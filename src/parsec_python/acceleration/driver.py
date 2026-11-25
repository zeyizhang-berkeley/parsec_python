"""Readable public workflow for the additive accelerated architecture."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from threading import Lock, current_thread
from time import perf_counter
from typing import Callable

import numpy as np

from parsec_python.SCF.single_point import (
    complete_single_point as complete_reference_single_point,
    prepare_single_point as prepare_reference_single_point,
)
from parsec_python.SCF.pbc import (
    prepare_periodic_single_point as prepare_reference_periodic_single_point,
)
from parsec_python.Grid.pbc import uses_fortran_routines
from parsec_python.Hartree.variational import VariationalHartreeSolve, tail_charges
from parsec_python.V_ion import ion_ion_energy
from parsec_python.models import (
    PreparationTimings,
    SCFIteration,
    SinglePointInput,
)

from .SCF.single_point import (
    AcceleratedPreparedSinglePointSystem,
    run_scf as run_accelerated_scf,
)
from .Grid import build_cluster_grid_by_slabs, fast_grid_requested
from .Hartree.poisson import build_hartree_problem, solve_scipy_hartree
from .Symmetry.axis_reflection import _fast_maps_requested
from .SCF.symmetry_fields import scf_buffers_requested
# Imported here and not where it is read: the module binds ``require_cupy``
# when it is first imported, and a preparation may run under a double of it.
from .V_ion.cupy_ionic import ionic_gpu_count_setting, ionic_sum_grid_setting
from .V_ion.periodic_setup_fortran import oblique_split as periodic_oblique_split
from .V_ion.periodic_setup_fortran import reach_rule as periodic_reach_rule
from .backends.implicit_stencil import (
    FilterPlan,
    counted_filter_applications,
    pack_worker_setting,
)
from .backends.scipy import ScipyHamiltonianBackend
from .backends.selection import BackendSelection, resolve_backend
from .models import (
    AcceleratedSinglePointResult,
    BackendName,
    BackendUnavailableError,
    SymmetryMode,
)


_REFERENCE_CACHE_FORMAT = 1
# GPU boundary builders, by class name: the import needs CuPy.
_GPU_BOUNDARY_BUILDERS = frozenset(
    {
        "CuPyMultipoleBoundaryBuilder",
        "CuPySymmetryMultipoleBoundaryBuilder",
        "CuPyPointMultipoleBoundaryBuilder",
        "CuPySubgroupMultipoleBoundaryBuilder",
    }
)
# The boundary of the resident chain on an axis-reflection wedge takes the
# wedge kernels whatever the plan (A100, 2026-11-07: O2 PBE in sectors, Hartree
# 1.12 s with the orbit table, 1.01 s with the full-grid kernels and 0.78 s
# with the wedge kernels, peaks 1074, 1230 and 1102 MiB).
_RESIDENT_WEDGE_KERNELS = True
_REFERENCE_CACHE: OrderedDict[str, object] = OrderedDict()
_REFERENCE_CACHE_LOCK = Lock()


def _resident_reference_cache_size() -> int:
    """Return the bounded resident static-system cache capacity."""

    if os.environ.get("PARSEC_ACCELERATED_RESIDENT", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return 0
    raw = os.environ.get("PARSEC_RESIDENT_REFERENCE_CACHE_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE must be an integer"
        ) from error
    if value < 0:
        raise ValueError(
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE cannot be negative"
        )
    return value


def _reference_cache_key(
    problem: SinglePointInput,
    selection: BackendSelection,
    *,
    defer_native_laplacian: bool,
    cache_directory: os.PathLike[str] | str | None,
) -> str:
    """Hash every physical/static input used by reference preparation."""

    digest = sha256()
    digest.update(f"resident-reference-v{_REFERENCE_CACHE_FORMAT}".encode("ascii"))
    digest.update(selection.finite_difference_builder.encode("ascii"))
    digest.update(
        (
            "cupy"
            if getattr(selection, "selected", None) == "cupy"
            and _gpu_ionic_requested(periodic=getattr(problem, "periodic_cell", None) is not None)
            else "native"
        ).encode("ascii")
    )
    digest.update(os.environ.get("PARSEC_NATIVE_PROJECTOR_LOOKUP", "1").encode("ascii"))
    # The physical reach of the periodic radial sums (the default since 2026-10-30) changes their
    # last bits: its key differs from that of the table reach, which adds nothing (the key of a cell
    # before 2026-10-30, when the table reach was the default).
    if getattr(problem, "periodic_cell", None) is not None and periodic_reach_rule() != "table":
        digest.update(f"periodic-reach-{periodic_reach_rule()}".encode("ascii"))
    # PARSEC_PERIODIC_OBLIQUE_SPLIT other than auto (the split of the orthorhombic grid on the oblique grid where a
    # species uses Cubic_Spline; the default up to 2026-11-22, which adds nothing: the default validated adds its name).
    if getattr(problem, "periodic_cell", None) is not None and periodic_oblique_split() != "auto":
        digest.update(f"periodic-oblique-{periodic_oblique_split()}".encode("ascii"))
    # PARSEC_PROJECTOR_TAIL other than its default auto (the projectors of the tables with a tail at rcore)
    from ..Pseudopotential.potre import PROJECTOR_TAIL_DEFAULT, projector_tail_setting

    if projector_tail_setting() != PROJECTOR_TAIL_DEFAULT:
        digest.update(f"projector-tail-{projector_tail_setting()}".encode("ascii"))
    # PARSEC_SPLINE_COULOMB_TAIL other than off (the spline of a local potential hands over to -2 Z/r at its Coulomb
    # onset), where a local potential can be a spline: a species with Cubic_Spline, or a cell (the default routines take
    # the spline in any case).  off, the rule before 2026-11-23, and the linear tables of a cluster keep the key they had
    from ..Pseudopotential.potre import spline_coulomb_tail_setting

    if spline_coulomb_tail_setting() != "off" and (
        getattr(problem, "periodic_cell", None) is not None
        or any(getattr(specification, "use_spline", False) for specification in problem.pseudopotentials.values())
    ):
        digest.update(f"spline-coulomb-tail-{spline_coulomb_tail_setting()}".encode("ascii"))
    digest.update(bytes((int(bool(defer_native_laplacian)),)))
    cache_path = (
        "disabled"
        if cache_directory is None
        else str(Path(cache_directory).resolve())
    )
    digest.update(cache_path.encode("utf-8"))
    for settings in (
        problem.grid,
        problem.scf,
        problem.hartree,
        problem.eigensolver,
        problem.mixing,
        problem.initial_density_settings,
        problem.recenter_geometry,
        problem.periodic_cell,
    ):
        digest.update(repr(settings).encode("utf-8"))
    initial_source = problem.initial_density_settings.file
    if initial_source is not None:
        source_path = Path(initial_source).resolve()
        digest.update(str(source_path).encode("utf-8"))
        if source_path.is_file():
            with source_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    model_checkpoint = problem.initial_density_settings.checkpoint
    if model_checkpoint is not None:
        checkpoint_path = Path(model_checkpoint).resolve()
        digest.update(str(checkpoint_path).encode("utf-8"))
        if checkpoint_path.is_file():
            checkpoint_stat = checkpoint_path.stat()
            digest.update(
                np.asarray(
                    [checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns],
                    dtype=np.int64,
                ).tobytes()
            )
    if (
        problem.initial_density_settings.method not in {"sad", "file"}
        and problem.initial_density_settings.file is None
    ):
        from parsec_python.MLDensity.providers import provider_source_fingerprint

        digest.update(
            provider_source_fingerprint(problem.initial_density_settings)
        )
    for atom in problem.atoms:
        digest.update(atom.symbol.encode("utf-8"))
        digest.update(
            np.ascontiguousarray(atom.position, dtype=np.float64).tobytes()
        )
    for symbol, specification in sorted(problem.pseudopotentials.items()):
        path = Path(specification.path).resolve()
        digest.update(symbol.encode("utf-8"))
        digest.update(str(path).encode("utf-8"))
        digest.update(np.int64(specification.local_angular_momentum).tobytes())
        digest.update(bytes((
            int(bool(specification.read_valence_density)),
            int(bool(specification.use_spline)),
        )))
        # Old_Interpolation_Format, Fourier filters: nothing at their defaults.
        from .V_ion.setup_options import hash_pseudopotential_setup

        hash_pseudopotential_setup(digest, specification)
        if getattr(specification, "has_hubbard", False):
            # the shape of the LDA+U projectors (PARSEC_LDAU_SMOOTH_PROJECTORS); the projectors of upot.F90 (off) add
            # nothing, the key of before
            from parsec_python.Hamiltonian.hubbard import ldau_smooth_projectors_fraction

            fraction = ldau_smooth_projectors_fraction()
            if fraction:
                digest.update(f"ldau-smooth-{fraction!r}".encode("ascii"))
        if specification.pseudopotential_format != "martins_new":
            # What the input adds to an old Martins file: the occupations,
            # hence the ionic charge, and the radius of the projectors.
            # Nothing is hashed for a new-format species, whose key stays
            # what it was.
            digest.update(repr((
                specification.pseudopotential_format,
                specification.number_of_channels,
                specification.electrons_per_orbital,
                specification.core_cutoff_radius,
            )).encode("utf-8"))
            if specification.pseudopotential_format == "fhipp":
                # Whether the core density of the FHIPP file is read.
                digest.update(
                    bytes((int(bool(specification.nonlinear_core_correction)),))
                )
                # PARSEC_FHI_ORIGIN=auto (samples below the first one of the file); off keeps the key of before
                from ..Pseudopotential.fhipp import fhi_origin_setting

                if fhi_origin_setting() != "off":
                    digest.update(f"fhi-origin-{fhi_origin_setting()}".encode("ascii"))
        if getattr(problem.scf, "spin_polarized", False):
            # The starting polarization of the species shapes the starting
            # spin densities of the prepared system.  Nothing is hashed
            # without spin polarization, whose key stays what it was.
            digest.update(
                np.float64(specification.initial_spin_polarization).tobytes()
            )
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _setup_option_details(problem, selection) -> tuple[tuple[str, str], ...]:
    """Backend details of the pseudopotential set-up options (:mod:`.V_ion.setup_options`)."""

    from .V_ion.setup_options import setup_option_details

    return setup_option_details(problem, selection)


def _cached_reference_view(
    reference,
    *,
    cache_directory: os.PathLike[str] | str | None,
    lookup_seconds: float,
):
    """Return fresh timing/descriptor metadata over immutable cached arrays.

    A deferred Laplacian gets a descriptor of its own for this calculation's
    report.  The full-grid matrix, once a calculation has had to build it, is
    shared with the cached descriptor and so with every later one.
    """

    negative_laplacian = reference.negative_laplacian
    from .Laplacian import DeferredNativeNegativeLaplacian, DeferredPeriodicNegativeLaplacian

    if isinstance(negative_laplacian, DeferredPeriodicNegativeLaplacian):
        # the descriptor of a periodic box stays one (its wrapped matrix, its rows from the stencil)
        negative_laplacian = DeferredPeriodicNegativeLaplacian(
            negative_laplacian.grid, sharing_matrix_with=negative_laplacian
        )
        negative_laplacian.reference_static_cache_status = "hit"
        negative_laplacian.reference_static_cache_lookup_seconds = lookup_seconds
    elif isinstance(negative_laplacian, DeferredNativeNegativeLaplacian):
        negative_laplacian = DeferredNativeNegativeLaplacian(
            negative_laplacian.grid,
            cache_directory=cache_directory,
            sharing_matrix_with=negative_laplacian,
        )
        negative_laplacian.reference_static_cache_status = "hit"
        negative_laplacian.reference_static_cache_lookup_seconds = lookup_seconds
    return replace(
        reference,
        negative_laplacian=negative_laplacian,
        timings=PreparationTimings(total_seconds=lookup_seconds),
    )


def _cupy_installed() -> bool:
    """Whether CuPy could be imported, decided without importing it."""

    if "cupy" in sys.modules:
        return True
    try:
        return find_spec("cupy") is not None
    except (ImportError, ValueError):
        return False


def _start_compile_ahead(problem, backend) -> bool:
    """Compile the kernels of ``problem`` ahead on background threads where the backend may be CuPy
    (``PARSEC_CUPY_COMPILE_AHEAD``, :mod:`.backends.compile_ahead`; ``ValueError`` for a value it does not know).

    Not on the CPU (``PARSEC_DEVICE=cpu``), for a host backend or where CuPy is not installed.
    """

    from .backends.compile_ahead import compile_ahead_setting, start

    if compile_ahead_setting() == "off":
        return False
    if str(backend).strip().lower() not in {"auto", "cupy"}:
        return False
    if os.environ.get("PARSEC_DEVICE", "auto").strip().lower() == "cpu" or not _cupy_installed():
        return False
    return start(problem)


def _reference_with_matrix(reference):
    """Replace a deferred Laplacian by its matrix for a backend that reads it.

    Only the CuPy backend understands the deferred descriptor.  The build
    time joins the finite-difference stage the matrix was deferred from.
    """

    from .Laplacian import DeferredNativeNegativeLaplacian

    operator = getattr(reference, "negative_laplacian", None)
    if not isinstance(operator, DeferredNativeNegativeLaplacian):
        return reference
    built_before = operator.materialization_seconds
    matrix = operator.materialize()
    seconds = operator.materialization_seconds - built_before
    return replace(
        reference,
        negative_laplacian=matrix,
        timings=replace(
            reference.timings,
            finite_difference_seconds=(
                reference.timings.finite_difference_seconds + seconds
            ),
            total_seconds=reference.timings.total_seconds + seconds,
        ),
    )


def _remember_reference(cache_key: str, reference, capacity: int) -> None:
    """Insert one immutable prepared system into the bounded process cache."""

    if capacity < 1:
        return
    with _REFERENCE_CACHE_LOCK:
        _REFERENCE_CACHE[cache_key] = reference
        _REFERENCE_CACHE.move_to_end(cache_key)
        while len(_REFERENCE_CACHE) > capacity:
            _REFERENCE_CACHE.popitem(last=False)


class _PreparationStages:
    """When every stage of one preparation began and ended.

    Seconds from the entry of :func:`prepare_single_point`, with the name of
    the thread that ran the stage.  Threads work beside each other and a
    stage may lie inside another of its thread (the grid inside the
    reference preparation), so the records are a timeline and not a stack:
    their seconds must not be added up.  A wait at a join is a stage of the
    thread that waits.
    """

    def __init__(self) -> None:
        self.origin = perf_counter()
        self._lock = Lock()
        self._records: list[tuple[str, str, float, float]] = []

    def add(self, name: str, started: float, ended: float | None = None) -> None:
        """Record stage ``name`` of the calling thread, begun at ``started``.

        ``started`` and ``ended`` are readings of ``perf_counter``; the
        stage ends now where ``ended`` is omitted.
        """

        ended = perf_counter() if ended is None else ended
        record = (
            name,
            current_thread().name,
            started - self.origin,
            ended - self.origin,
        )
        with self._lock:
            self._records.append(record)

    def timed(self, name, function):
        """Return ``function`` with every call recorded as stage ``name``.

        ``name`` may be a function of the arguments of the call.  A call
        that raises is recorded up to the failure.
        """

        def stage(*arguments, **options):
            started = perf_counter()
            try:
                return function(*arguments, **options)
            finally:
                self.add(
                    name(*arguments, **options) if callable(name) else name,
                    started,
                )

        return stage

    def records(self) -> tuple[dict[str, object], ...]:
        """The stages in the order of their start."""

        with self._lock:
            records = sorted(self._records, key=lambda record: record[2:])
        return tuple(
            dict(stage=name, thread=thread, start_seconds=start, end_seconds=end)
            for name, thread, start, end in records
        )


class _Published:
    """One value that a thread of a preparation hands to the others, once.

    A reader waits until the value is there.  Whoever may leave a reader
    waiting publishes ``None`` on its way out, so that no thread waits for
    a stage that has failed or was never reached.
    """

    def __init__(self) -> None:
        self._future: Future = Future()
        self._lock = Lock()

    def publish(self, value) -> None:
        """Hand ``value`` to the readers unless a value is there already."""

        with self._lock:
            if not self._future.done():
                self._future.set_result(value)

    def wait(self):
        """Return the value, once it has been published."""

        return self._future.result()


def _symmetry_overlap_requested() -> bool:
    """``PARSEC_OVERLAP_SYMMETRY_SETUP``: on unless 0, false, no or off.

    On, a rank detects its symmetry and builds the representation phases on
    a thread that starts when the grid exists, beside the projectors, where
    its reference preparation builds no ionic field (see
    :class:`_OverlappedSymmetryMaps`).  Off, both follow the reference
    preparation on the thread that prepares, as before.  The maps are the
    values of the same functions of the same grid and atoms either way.
    """

    return os.environ.get(
        "PARSEC_OVERLAP_SYMMETRY_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _stabilizer_symmetry_disabled() -> bool:
    """Parse ``PARSEC_ORBITAL_STABILIZER_SYMMETRY``: whether it is off."""

    policy = os.environ.get(
        "PARSEC_ORBITAL_STABILIZER_SYMMETRY", "auto"
    ).strip().lower()
    if policy not in {
        "auto", "on", "off", "1", "0", "true", "false", "yes", "no"
    }:
        raise ValueError(
            "PARSEC_ORBITAL_STABILIZER_SYMMETRY must be auto, on, or off"
        )
    return policy in {"off", "0", "false", "no"}


def single_sector_setting() -> str:
    """``PARSEC_CUPY_SINGLE_SECTOR``: ``auto`` (default), ``on`` or ``off``.

    ``on`` gives a CuPy calculation without point-group symmetry one sector
    of the identity; ``auto`` does so in the sector context of the full-SCF
    runner (molecular dynamics, structures without symmetry), and leaves the
    other routes as they were.  ``off`` never does.  The numbers are those
    of the full grid.  The basis of that sector is shared where it is given
    a group: by the MPI ranks of the runner, or by the GPUs of its node
    where it runs one rank, and in ``main.py`` by the GPUs of the process
    (``PARSEC_CUPY_SECTOR_PLACEMENT=size``, the default since 2026-11-19,
    ``Eigensolvers/symmetry.process_sector_device_groups``) where a CHEBFF
    first solve shares it (``distributed_state.shared_basis_devices``).
    With ``PARSEC_CUPY_SECTOR_PLACEMENT=cyclic`` the basis lies on the first
    GPU of the process, as the full grid did (measured on four A100,
    2026-11-18).
    """

    value = os.environ.get("PARSEC_CUPY_SINGLE_SECTOR", "auto").strip().lower() or "auto"
    if value in {"1", "true", "yes"}:
        value = "on"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in {"auto", "on", "off"}:
        raise ValueError("PARSEC_CUPY_SINGLE_SECTOR must be auto, on or off")
    return value


def spin_cell_group_refusal(reason: str) -> str:
    """The refusal of a spin-polarized periodic cell that would reach an MPI SCF without a sector group.

    The ranks of the MPI runner solve both channels of a spin cell in the abelian sectors of its group
    (``PARSEC_GAMMA_SPIN_SECTORS=auto``); the one sector of the identity, whose basis the ranks share, takes both
    channels only for the forces, relaxations and molecular dynamics that a run takes without a group (since
    2026-11-21: ``--symmetry off``, ``spin_identity_sector``, ``PARSEC_SPIN_SECTOR_FORCES=auto``, with
    ``Use_Fortran_Routines: .false.``; ``single_sector`` below needs an unpolarized run).  ``reason`` says why this
    cell has no such group.  The runner asks this before its preparation where it knows (no symmetry, ``PARSEC_PERIODIC_SYMMETRY=off``), the
    driver before the reference preparation in the same cases and after the detection otherwise (since
    2026-11-19; before, the message named ``PARSEC_CUPY_SINGLE_SECTOR``, which cannot help).
    """

    return (
        "a spin-polarized periodic cell runs as an MPI SCF only in the abelian sectors of its group, both channels "
        f"divided among the ranks, and this cell has none: {reason}. Its forces, relaxations and molecular dynamics "
        "with Use_Fortran_Routines: .false. run on the ranks with --symmetry off, both channels as one shared "
        "sector of the identity (PARSEC_SPIN_SECTOR_FORCES=auto); "
        "a single point runs with parsec_python/main.py, whose one process solves both channels on the full grid of "
        "one device, or with --serial-control"
    )


def periodic_sector_refusal(problem, *, hybrid_sectors: bool = False, features: bool = False) -> str | None:
    """Why a periodic cell cannot be solved in the sectors of the Gamma route, or ``None``.

    The route of the sectors takes a bulk cell at the Gamma point with real
    orbitals: an orthorhombic cell, unpolarized or with collinear spin
    (``PARSEC_GAMMA_SPIN_SECTORS=auto``: both channels in the abelian
    sectors, :mod:`.SCF.gamma_spin`), with a native LDA or GGA, or in one
    process a meta-GGA, a libxc functional, LDA+U (unpolarized, and with
    spin under ``PARSEC_LDAU_SPIN_SECTORS=auto``) or the TS correction under
    their switches (``features`` below); in one process also a wire or slab under ``PARSEC_SLAB_WIRE_SECTORS=auto``
    (:func:`_wire_slab_sector_refusal`; the default since 2026-11-23, where its sectors meet the rule of
    :func:`_wire_slab_sector_shortfall`) and a cell whose lattice vectors are not along the axes under
    ``PARSEC_OBLIQUE_SECTORS=auto`` (:func:`_oblique_sector_refusal`, the default since 2026-11-23).
    Its orbitals are solved in the sectors of the point group of the cell
    about its origin (``PARSEC_PERIODIC_SYMMETRY=auto``, the default), whose
    density, potentials and mixer keep one value per orbit, the Hartree
    potential solved on the full grid of the cell and reduced again; or, for
    a cell without a group or with ``PARSEC_PERIODIC_SYMMETRY=off``, as one
    sector of the identity, whose basis the MPI ranks of the full-SCF runner
    share (in one process, ``PARSEC_CUPY_SINGLE_SECTOR=on``: the GPUs of the
    process share it under ``PARSEC_CUPY_SECTOR_PLACEMENT=size``), every
    field on the full grid.  The route has not been built or
    validated for the other periodic calculations named here; a calculation
    that names one keeps the full grid of one device in ``main.py`` (a
    spin-polarized cell goes to the k-point driver, whose k-point sectors may
    still split the Gamma point: ``SCF/kpoint_sectors.py``; so does, under
    ``PARSEC_GAMMA_HYBRID_SECTORS=kpoint``, an unpolarized hybrid of a cell
    large enough for those sectors, ``SCF/gamma_hybrid_route.py``), and the
    MPI runner refuses it (but the SCF of an unpolarized hybrid, below).

    ``hybrid_sectors``: the caller solves an unpolarized hybrid in the real
    sectors of this route (the driver in one process under
    ``PARSEC_GAMMA_HYBRID_SECTORS=auto``, :func:`_gamma_hybrid_in_sectors`):
    a hybrid is then no reason; the ranks of the MPI runner pass it where they
    take the SCF of an unpolarized hybrid (:func:`_mpi_periodic_hybrid`, since
    2026-11-22).  The k-point mesh folded into a supercell keeps refusing it.

    ``features``: the caller runs the cell of the input in one process (the
    driver without an MPI context, the spin route of the command line), where
    the Tkatchenko-Scheffler correction, LDA+U (a spin cell too, from 2^26
    grid points x states under ``PARSEC_LDAU_SPIN_SECTORS=auto``), a meta-GGA
    and a libxc functional take the sectors under their switches
    (``PARSEC_GAMMA_TS_SECTORS``, ``PARSEC_GAMMA_LDAU_SECTORS``,
    ``PARSEC_GAMMA_META_GGA_SECTORS``, ``PARSEC_GAMMA_LIBXC_SECTORS``:
    :mod:`.SCF.gamma_features`); not with a hybrid.  The other callers keep
    refusing them.
    """

    from parsec_python.Grid import kpoint_sampling

    scf = problem.scf
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    if int(getattr(problem.grid, "periodicity", 3)) != 3:
        return _wire_slab_sector_refusal(problem, features=features)
    if problem.has_kpoint_sampling:
        sampling = kpoint_sampling(problem)
        if sampling is None or len(sampling[0]) != 1 or np.any(sampling[0]):
            return "k-point sampling beyond the Gamma point"
    if scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit:
        # Collinear spin: both channels in the abelian sectors of the cell (PARSEC_GAMMA_SPIN_SECTORS=auto,
        # SCF/gamma_spin.py); off refuses it as before.
        from .SCF.gamma_spin import gamma_spin_sector_refusal

        spin_refusal = gamma_spin_sector_refusal(problem)
        if spin_refusal is not None:
            return spin_refusal
    if scf.is_hybrid and not hybrid_sectors:
        return "a hybrid functional"
    # TS, LDA+U, meta-GGA and libxc: the sectors where the caller runs one process on the cell of the input
    # (``features``) and the switch of each takes the cell (SCF/gamma_features.py); refused as before otherwise.
    from .SCF.gamma_features import correction_refusal, meta_gga_or_libxc_refusal

    functional_refusal = meta_gga_or_libxc_refusal(problem, features=features)
    if functional_refusal is not None:
        return functional_refusal
    if not np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10):
        oblique_refusal = _oblique_sector_refusal(problem, features=features)
        if oblique_refusal is not None:
            return oblique_refusal
    return correction_refusal(problem, features=features)


_OBLIQUE_REASON = "an oblique cell (lattice vectors not along the Cartesian axes)"


def _oblique_sector_refusal(problem, *, features: bool = False) -> str | None:
    """Why a bulk cell whose lattice vectors are not along the Cartesian axes cannot be solved in the sectors, or
    ``None``.

    ``PARSEC_OBLIQUE_SECTORS=auto`` takes such a cell at the Gamma point in one process (``features``), unpolarized,
    with a native LDA or GGA and none of TS, LDA+U, a hybrid, a meta-GGA or libxc: its orbitals in the sectors of the
    operations that map its grid and finite-difference operators onto themselves, detected in the frame of the grid
    indices (``Symmetry/oblique_frame.py``; ``auto`` is the default since 2026-11-23, and the floor of
    ``PARSEC_PERIODIC_SECTOR_FLOOR`` applies as to other cells).  Everything else, and every such cell under ``off``,
    keeps the full grid as before.  Its forces and the moves of a relaxation are averaged over the Cartesian operations
    ``S W S^-1`` of the sectors (:mod:`.Forces.sector_symmetry`).
    """

    from .Symmetry.oblique_frame import oblique_sectors_setting

    if oblique_sectors_setting() != "auto" or not features:
        return _OBLIQUE_REASON
    if int(getattr(problem.periodic_cell, "dimensions", 3)) != 3:
        # the slab of the default routines (Use_Fortran_Routines: .false.): its grid keeps the full grid
        return _OBLIQUE_REASON
    scf = problem.scf
    if scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit:
        return f"{_OBLIQUE_REASON} with spin polarization or spin-orbit coupling"
    if scf.is_hybrid:
        return "a hybrid functional"
    from .SCF.gamma_features import correction_refusal, meta_gga_or_libxc_refusal

    functional_refusal = meta_gga_or_libxc_refusal(problem, features=False)
    if functional_refusal is not None:
        return functional_refusal
    return correction_refusal(problem, features=False)


def _symmetry_index_frame(grid, problem, mpi_context=None):
    """The :class:`.Symmetry.oblique_frame.IndexFrame` the detection of the sectors of ``grid`` takes, or ``None``
    (a grid of the Cartesian axes, ``PARSEC_OBLIQUE_SECTORS=off``, or a calculation that :func:`_oblique_sector_refusal`
    refuses: an MPI run, spin, a hybrid, TS, LDA+U, a meta-GGA or libxc, also where the lattice is diagonal and only
    the general grid of the Fortran-derived routines asks for the frame).  Forces and moves averaged over the group of
    the sectors (``sector_forces``) take the frame too: the operations turn the forces as ``S W S^-1``
    (:mod:`.Forces.sector_symmetry`)."""

    from .Symmetry.oblique_frame import index_frame, needs_index_frame, oblique_sectors_setting

    if not needs_index_frame(grid) or oblique_sectors_setting() != "auto":
        return None
    if _oblique_sector_refusal(problem, features=mpi_context is None) is not None:
        return None
    return index_frame(grid, general=_oblique_full_group(problem))


def _oblique_full_group(problem) -> bool:
    """Whether the frame of an oblique bulk cell also takes the integer operations of its grid indices that are no
    signed permutation (``PARSEC_OBLIQUE_FULL_GROUP=auto``, :func:`.Symmetry.oblique_frame.general_index_operations`):
    a functional without a density gradient (:data:`.Symmetry.oblique_frame.GRADIENT_FREE_FUNCTIONALS`) at the Gamma
    point of a bulk cell, whose discrete Hamiltonian commutes with them (the gradient of a GGA and the k-point term do
    not)."""

    from .Symmetry.oblique_frame import GRADIENT_FREE_FUNCTIONALS, oblique_full_group_setting

    if oblique_full_group_setting() != "auto":
        return False
    cell = getattr(problem, "periodic_cell", None)
    if (
        cell is None
        or int(getattr(cell, "dimensions", 3)) != 3
        or int(getattr(problem.grid, "periodicity", 3)) != 3
    ):
        return False
    if str(getattr(problem.scf, "xc_functional", "")).strip().lower() not in GRADIENT_FREE_FUNCTIONALS:
        return False
    if getattr(problem, "has_kpoint_sampling", False):
        from parsec_python.Grid import kpoint_sampling

        sampling = kpoint_sampling(problem)
        if sampling is None or len(sampling[0]) != 1 or np.any(sampling[0]):
            return False
    return True


def _index_frame_cell(problem) -> bool:
    """Whether the sectors of this bulk cell are those of the frame of its grid indices (``PARSEC_OBLIQUE_SECTORS``,
    :func:`.Symmetry.oblique_frame.needs_index_frame`), told from the input before its grid is built: lattice vectors
    not along the Cartesian axes, or the general grid of the Fortran-derived routines (sides of different steps).

    Below the floor of ``PARSEC_PERIODIC_SECTOR_FLOOR`` such a cell keeps the full grid for its forces and moves too
    (the default ``auto`` since 2026-11-23): the full grid it had at every size before, where the forces of a cell of
    up to about 8 x 10^6 grid points x states took 0.5-0.9 s less on a first run of the command line (A100) than its
    sectors.  ``False`` for a wire or slab and for a cluster."""

    cell = getattr(problem, "periodic_cell", None)
    if (
        cell is None
        or int(getattr(problem.grid, "periodicity", 3)) != 3
        or int(getattr(cell, "dimensions", 3)) != 3
    ):
        return False
    lattice = np.asarray(cell.lattice_vectors, dtype=np.float64)
    if not np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10):
        return True
    if not getattr(problem.grid, "fortran_routines", False):
        return False
    from parsec_python.Grid.pbc import uses_general_periodic_grid

    try:
        return bool(uses_general_periodic_grid(cell, problem.grid))
    except (AttributeError, TypeError, ValueError):
        return False


_WIRE_SLAB_REASON = "a wire or slab (Boundary_Conditions: wire or slab)"


def _wire_slab_sector_refusal(problem, *, features: bool = False) -> str | None:
    """Why a wire or slab cannot be solved in the sectors of the Gamma route, or ``None``.

    ``PARSEC_SLAB_WIRE_SECTORS=auto`` takes a wire or slab of the Fortran-derived routines (the grid of
    :mod:`parsec_python.Grid.confined`) at the Gamma point whose periodic vectors lie along the Cartesian axes,
    unpolarized, with a native LDA or GGA and none of TS, LDA+U, a hybrid, a meta-GGA, libxc or
    ``Symmetrize_Charge_Density``, in one process (``features``: the driver without an MPI context): its orbitals in
    the sectors of its point group about the origin (``Symmetry/axis_reflection.py``), every field on the full grid of the
    wire or slab reduced to the orbits, its Hartree potential solved there by its own solver (the boundary values of
    ``hartset_wire``/``hartset_slab``).  ``auto`` is the default since 2026-11-23; under
    ``PARSEC_PERIODIC_SECTOR_FLOOR=auto`` a wire or slab keeps the full grid where its solved sectors are fewer than
    :data:`WIRE_SLAB_MINIMUM_SECTORS` or smaller than :data:`WIRE_SLAB_SECTOR_ENTRIES` grid points times states
    (:func:`_cell_sector_floor` before the detection, :func:`_wire_slab_sector_shortfall` after it).  Everything
    else, and every wire or slab under ``off``, keeps the full grid as before.  The forces of a wire or slab stay with the reference solver
    (:func:`.SCF.periodic_routes.moving_cell_options_reference_only`), its MPI runs and restarts too
    (:func:`.SCF.periodic_routes.wire_slab_reference_only`).
    """

    from .Symmetry.axis_reflection import slab_wire_sectors_setting

    if slab_wire_sectors_setting() != "auto" or not features:
        return _WIRE_SLAB_REASON
    from parsec_python.Grid import kpoint_sampling

    scf = problem.scf
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    if not bool(getattr(problem.grid, "fortran_routines", True)):
        return _WIRE_SLAB_REASON
    if not np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10):
        # (a hexagonal slab: its sectors in the frame of its grid indices under PARSEC_OBLIQUE_SECTORS=auto)
        from .Symmetry.oblique_frame import oblique_sectors_setting

        if oblique_sectors_setting() != "auto":
            return "a wire or slab whose periodic vectors are not along the Cartesian axes (PARSEC_OBLIQUE_SECTORS=off)"
    if problem.has_kpoint_sampling:
        sampling = kpoint_sampling(problem)
        if sampling is None or len(sampling[0]) != 1 or np.any(sampling[0]):
            return "k-point sampling beyond the Gamma point"
    if scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit:
        return "a wire or slab with spin polarization or spin-orbit coupling"
    if scf.is_hybrid:
        return "a hybrid functional"
    if bool(getattr(scf, "symmetrize_density", False)):
        return "a wire or slab with Symmetrize_Charge_Density"
    from .SCF.gamma_features import correction_refusal, meta_gga_or_libxc_refusal

    functional_refusal = meta_gga_or_libxc_refusal(problem, features=False)
    if functional_refusal is not None:
        return functional_refusal
    return correction_refusal(problem, features=False)


def _gamma_hybrid_in_sectors(problem, mpi_context, *, sector_forces: bool = False) -> bool:
    """Whether an unpolarized hybrid of a bulk cell at the Gamma point takes the sectors of the Gamma route here.

    ``PARSEC_GAMMA_HYBRID_SECTORS=auto`` (the default, ``SCF/gamma_hybrid_route.py``) in one process: the orbitals
    in the real sectors of the abelian group of the cell (its commuting involutions with the half-cell
    translations; a hybrid keeps no weighted sectors, ``_orbital_group_refusal``), the exchange of the device Fock
    loop from their orbitals unfolded to the full grid, where the cuFFT pair solver of the cell solves the pair
    densities, and its compressed operator folded to every sector (``Hamiltonian/exchange_device.py``).  A cell
    whose basis (grid points times states) is below the floor of the k-point sectors (``2^24`` entries,
    ``gamma_hybrid_route.sector_basis_reached``) keeps the full grid, where the sectors gain nothing (an 8-atom
    cube took 1.8 times longer in them).  With ``sector_forces`` (its caller averages the forces and moved positions
    over the group of the sectors, :func:`prepare_single_point`) such a cell takes the sectors whatever its size
    under ``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=sectors`` (an opt-in, slower on small cells); under ``auto`` (the
    default) it keeps the full grid, whose forces and moves are averaged over the operations of the cell
    (:func:`_gamma_hybrid_full_grid_forces`), and under ``off`` it keeps the full grid as before.  ``off`` and
    ``kpoint`` of ``PARSEC_GAMMA_HYBRID_SECTORS`` keep the former routes; with an MPI context it is ``False``: the
    ranks of the MPI runner decide with :func:`_mpi_periodic_hybrid` (since 2026-11-22; before, they refused every
    periodic hybrid).
    """

    from .SCF.gamma_hybrid_route import (
        gamma_hybrid_sector_forces_setting,
        gamma_hybrid_sectors_setting,
        sector_basis_reached,
    )

    if not (
        mpi_context is None
        and bool(getattr(getattr(problem, "scf", None), "is_hybrid", False))
        and gamma_hybrid_sectors_setting() == "auto"
    ):
        return False
    if sector_basis_reached(problem):
        return True
    return bool(sector_forces) and gamma_hybrid_sector_forces_setting() == "sectors"


def _mpi_periodic_hybrid(problem, mpi_context) -> bool | None:
    """How the ranks of the MPI full-SCF runner solve an unpolarized hybrid of a bulk cell at the Gamma point, or
    ``None`` where the run is not one (since 2026-11-22; before, the runner refused every periodic hybrid).

    ``True``: in the real sectors of the abelian group of the cell, as one process takes them under
    ``PARSEC_GAMMA_HYBRID_SECTORS=auto`` (:func:`_gamma_hybrid_in_sectors`, the same floor of
    :func:`.SCF.gamma_hybrid_route.sector_basis_reached`), every sector basis on one GPU of its rank, the exchange
    solved by the cuFFT pair solver of the cell (:mod:`.SCF.hybrid_mpi`): since 2026-11-23 in the coset blocks of
    the half-cell translations of the sectors or (from 32 sectors) in one full-grid block, gathered on every rank, a
    small build whole on each rank and a larger one divided among the GPUs of all ranks
    (``PARSEC_MPI_EXCHANGE_POOL=auto``; before, and under ``off``, the pair tiles of the sectors on every GPU of every
    rank).  ``False``: as the one sector of the identity, on the first GPU of the root, whose small builds that GPU
    forms alone (since 2026-11-23, as one process does) and whose other builds every GPU of every rank solves in
    tiles: a cell below that floor, and every cell under ``off``, the full grid that one process takes there.
    ``kpoint`` (a route of one process) gives ``None``, and the hybrid is refused as before.  A spin-polarized hybrid
    of a cell gives ``None`` as well.  The input alone decides, alike on every rank.
    """

    scf = getattr(problem, "scf", None)
    if (
        mpi_context is None
        or getattr(problem, "periodic_cell", None) is None
        or not bool(getattr(scf, "is_hybrid", False))
        or bool(getattr(scf, "spin_polarized", False))
        or bool(getattr(scf, "noncollinear", False))
        or bool(getattr(scf, "self_consistent_spin_orbit", False))
    ):
        return None
    from .SCF.gamma_hybrid_route import gamma_hybrid_sectors_setting, sector_basis_reached

    setting = gamma_hybrid_sectors_setting()
    if setting == "kpoint":
        return None
    return setting == "auto" and bool(sector_basis_reached(problem))


def _gamma_hybrid_full_grid_forces(
    problem, mpi_context, *, sector_forces: bool, symmetry_mode: str, periodic_symmetry_mode: str
) -> bool:
    """Whether a Gamma-point hybrid kept on the full grid below the floor averages its forces and moved positions over
    the operations of the cell (``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto``, the default since 2026-11-21).

    The command line counts the forces and the relaxation by an optimizer of a bulk cell at the Gamma point as forces
    of the sectors (``sector_forces``, ``PARSEC_PERIODIC_SECTOR_FORCES=auto``), which it averages over the group of
    the sectors; a hybrid whose basis is below the floor of :func:`_gamma_hybrid_in_sectors` keeps the full grid,
    where nothing averaged them on the default (Grieder) routines.  This is ``True`` for such a cell: an unpolarized
    hybrid in one process under ``PARSEC_GAMMA_HYBRID_SECTORS=auto``, ``PARSEC_PERIODIC_SYMMETRY=auto`` and a
    symmetry mode that is not ``off``, on the default routines, that the route of the sectors would take above the
    floor (:func:`periodic_sector_refusal`).  Its forces and the moved positions of a relaxation are then averaged
    over the operations the Fortran-derived routines take on the full grid
    (:func:`parsec_python.Forces.periodic.periodic_force_operations`: the space group the problem carries, else the
    point operations about the origin that keep the lattice, the grid and the atoms), the SCF unchanged.  The
    Fortran-derived routines average over them already (``use_symmetry`` of the parser) and are left as they are.
    """

    from .SCF.gamma_hybrid_route import (
        gamma_hybrid_sector_forces_setting,
        gamma_hybrid_sectors_setting,
        sector_basis_reached,
    )

    scf = getattr(problem, "scf", None)
    if not (
        sector_forces
        and mpi_context is None
        and getattr(problem, "periodic_cell", None) is not None
        and bool(getattr(scf, "is_hybrid", False))
        and not bool(getattr(scf, "spin_polarized", False))
        and symmetry_mode != "off"
        and periodic_symmetry_mode == "auto"
        and gamma_hybrid_sectors_setting() == "auto"
        and gamma_hybrid_sector_forces_setting() == "auto"
        and not getattr(getattr(problem, "grid", None), "fortran_routines", False)
    ):
        return False
    if sector_basis_reached(problem):
        return False
    return periodic_sector_refusal(problem, hybrid_sectors=True, features=True) is None


_PERIODIC_FLOOR_SWITCH = "PARSEC_PERIODIC_SECTOR_FLOOR"


def periodic_sector_floor_setting() -> int:
    """``PARSEC_PERIODIC_SECTOR_FLOOR``: the smallest basis (grid points of the cell times states) whose Gamma-point
    SCF takes the sectors of the point group of the cell in one process.

    ``auto`` (the default): ``2^24`` entries, the floor of the k-point sectors and of the Gamma hybrids
    (``SCF/kpoint_sectors._MINIMUM_ENTRIES``); a smaller cell keeps the full grid of one device, where its SCF is
    faster (every sector costs an SCF step about 10 ms of fixed work, ``PARSEC_SECTOR_OVERHEAD``; re-measured on
    2026-11-21 after the setup of the sectors was cut, the crossover lies near 10^7 entries, README).  ``off``: no floor,
    every cell with a group takes the sectors, as before (to 2026-11-19); an integer: that floor.  An MPI run keeps
    its sectors whatever the size (its ranks share no full grid).
    """

    raw = os.environ.get(_PERIODIC_FLOOR_SWITCH, "auto").strip().lower() or "auto"
    if raw == "auto":
        from .SCF.kpoint_sectors import _MINIMUM_ENTRIES

        return int(_MINIMUM_ENTRIES)
    if raw == "off":
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{_PERIODIC_FLOOR_SWITCH} must be auto, off or a number of entries >= 0, not {raw!r}")
    return value


#: The rule of ``PARSEC_PERIODIC_SECTOR_FLOOR=auto`` for a wire or slab (``PARSEC_SLAB_WIRE_SECTORS=auto``): its
#: sectors are taken where they are at least :data:`WIRE_SLAB_MINIMUM_SECTORS` solved sectors of at least
#: :data:`WIRE_SLAB_SECTOR_ENTRIES` grid points of its cylinder or slab times states each
#: (:func:`_wire_slab_sector_shortfall`).  Each solved sector costs an SCF step about 12 ms of fixed work on A100,
#: which smaller sectors do not repay, and two sectors (a group of two operations) saved no time in the cells measured;
#: three weighted sectors of a group of four did (README, "Wires, slabs and oblique cells").
WIRE_SLAB_SECTOR_ENTRIES = 2_500_000
WIRE_SLAB_MINIMUM_SECTORS = 3
#: The floor of the basis that the rule above implies (no smaller wire or slab can meet it), applied before the
#: detection of the group (:func:`_cell_sector_floor`).
WIRE_SLAB_FLOOR_ENTRIES = WIRE_SLAB_SECTOR_ENTRIES * WIRE_SLAB_MINIMUM_SECTORS


def _wire_slab_floor_applies(problem) -> bool:
    """Whether ``problem`` is a wire or slab under ``PARSEC_PERIODIC_SECTOR_FLOOR=auto`` (the rule of
    :data:`WIRE_SLAB_SECTOR_ENTRIES`; an integer or ``off`` of the switch applies to it as to a bulk cell)."""

    if int(getattr(getattr(problem, "grid", None), "periodicity", 3)) not in (1, 2):
        return False
    return (os.environ.get(_PERIODIC_FLOOR_SWITCH, "auto").strip().lower() or "auto") == "auto"


def _cell_sector_floor(problem, floor: int) -> int:
    """The floor of the sectors of ``problem`` in one process before its group is known: ``floor``
    (``PARSEC_PERIODIC_SECTOR_FLOOR``), but :data:`WIRE_SLAB_FLOOR_ENTRIES` for a wire or slab under ``auto``, whose
    sectors :func:`_wire_slab_sector_shortfall` judges once they are found."""

    return WIRE_SLAB_FLOOR_ENTRIES if _wire_slab_floor_applies(problem) else floor


def _wire_slab_sector_shortfall(problem, grid, reduction, mpi_context, *, sector_forces: bool = False) -> str | None:
    """Why the sectors of ``reduction`` are not taken for a wire or slab in one process, or ``None``.

    Under ``PARSEC_PERIODIC_SECTOR_FLOOR=auto`` a wire or slab keeps the full grid where its solved sectors
    (:func:`_sector_count`) are fewer than :data:`WIRE_SLAB_MINIMUM_SECTORS` or hold fewer than
    :data:`WIRE_SLAB_SECTOR_ENTRIES` grid points times states each.  Nothing is refused for another cell, an MPI run,
    forces averaged over the group (``sector_forces``) or an integer or ``off`` value of the switch.
    """

    if reduction is None or mpi_context is not None or sector_forces or not _wire_slab_floor_applies(problem):
        return None
    from .SCF.gamma_hybrid_route import _states

    try:
        basis = int(grid.size) * int(_states(problem))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    sectors = max(1, _sector_count(reduction))
    if sectors >= WIRE_SLAB_MINIMUM_SECTORS and basis >= WIRE_SLAB_SECTOR_ENTRIES * sectors:
        return None
    return (
        f"the {sectors} solved sector{'s' if sectors != 1 else ''} of the wire or slab hold {basis // sectors} grid "
        f"points x states each; it takes its sectors from {WIRE_SLAB_MINIMUM_SECTORS} sectors of "
        f"{WIRE_SLAB_SECTOR_ENTRIES} or more ({_PERIODIC_FLOOR_SWITCH}=auto)"
    )


_PERIODIC_SPIN_FLOOR_SWITCH = "PARSEC_PERIODIC_SPIN_SECTOR_FLOOR"
#: The floor of ``PARSEC_PERIODIC_SPIN_SECTOR_FLOOR=auto``: grid points of the cell times states of a channel.  0, no
#: floor: the full grid of a small magnetic cell can need more SCF steps than ``Max_Iter`` allows where its sectors
#: converge (bcc Co2 at 1e-6 Ry: 24 steps in the sectors, not converged in the default 50 on the full grid), and no
#: size separates those cells from the others.
PERIODIC_SPIN_FLOOR_ENTRIES = 0


def periodic_spin_sector_floor_setting() -> int:
    """``PARSEC_PERIODIC_SPIN_SECTOR_FLOOR``: the smallest basis (grid points of the cell times states of a channel)
    whose collinear spin-polarized Gamma-point SCF takes the sectors of the point group of the cell in one process.

    ``auto`` (the default): :data:`PERIODIC_SPIN_FLOOR_ENTRIES`, no floor.  ``off``: no floor.  An integer: a smaller
    cell with a native LDA or GGA keeps the full grid of one device (2097152 suits the cells measured, README).  An
    MPI run, ``--symmetry on``, forces or moves averaged over the group of the sectors and an input whose species start
    with polarizations of opposite sign keep the sectors whatever the size.
    """

    raw = os.environ.get(_PERIODIC_SPIN_FLOOR_SWITCH, "auto").strip().lower() or "auto"
    if raw == "auto":
        return PERIODIC_SPIN_FLOOR_ENTRIES
    if raw == "off":
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{_PERIODIC_SPIN_FLOOR_SWITCH} must be auto, off or a number of entries >= 0, not {raw!r}")
    return value


def _periodic_sector_cost(problem, reference, mpi_context) -> float:
    """The fixed cost of one solved sector of a periodic cell in units of its grid points times states
    (``PARSEC_SECTOR_OVERHEAD``, :func:`.Symmetry.axis_reflection.sector_overhead_entries`), which the choice of its
    weighted plan adds per sector (:func:`.Symmetry.detect_point_group_reduction`); ``0`` (the choice of before) under
    ``off`` and in an MPI run, whose ranks solve their sectors beside each other and must divide them.

    Under ``auto`` only where that cost is at least the whole basis of the cell (grid points x states at most
    ``2^21``: the cells of a few atoms), where the fixed cost of the sectors decides their SCF; a larger cell keeps the
    plan of the least rows x states (its fewer, larger sectors save less than the SCF steps they may add: Si 2x2x2
    took 24 instead of 10 steps, README).  An integer value applies its cost to every unpolarized cell.

    ``0`` for a spin-polarized (or non-collinear, spin-orbit) run under every value: the guard of the spin sectors
    (:mod:`.SCF.spin_sectors`) sees a partly filled multiplet only where its partners lie in two or more sectors or
    in a state of weight above one, while the weighted sectors average each channel density over ``G``.  The plans of
    fewer sectors hold the partners of most multiplets of ``G`` as several states of one sector of weight 1 (bcc Fe2:
    the subgroup ``{E, t(1/2,1/2,1/2)}``, weights 1 and 1), where the guard cannot fire: spin-polarized bcc Fe2 at
    100 K stayed 0.125 Ry above the full grid, which the plan of the least rows x states reaches through its guard.
    """

    from .Symmetry.axis_reflection import sector_overhead_entries, sector_overhead_setting

    entries = sector_overhead_entries()
    if mpi_context is not None or entries <= 0:
        return 0.0
    scf = getattr(problem, "scf", None)
    if any(bool(getattr(scf, name, False)) for name in ("spin_polarized", "noncollinear", "self_consistent_spin_orbit")):
        # the plan of the least rows x states, whose sectors the spin guard was built for
        return 0.0
    from .SCF.gamma_hybrid_route import _states

    try:
        basis = int(reference.grid.size) * int(_states(problem))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0.0
    if basis <= 0:
        return 0.0
    cost = float(entries) / float(basis)
    if sector_overhead_setting() == "auto" and cost < 1.0:
        return 0.0
    return cost


def _periodic_basis_below_floor(
    problem, mpi_context, floor: int, *, sector_forces: bool = False
) -> tuple[int, int] | None:
    """``(rows, states)`` of a Gamma cell whose basis is below ``floor`` in one process, else ``None``.

    ``sector_forces``: the run computes forces or moves atoms averaged over the group of the sectors
    (``PARSEC_PERIODIC_SECTOR_FORCES=auto``, :func:`prepare_single_point`), which the full grid would not average:
    such a cell keeps its sectors whatever its size (``None``).  The rows are counted without building the grid
    (:func:`.SCF.gamma_hybrid_route.periodic_grid_points`).
    """

    if mpi_context is not None or floor <= 0 or sector_forces:
        return None
    from .SCF.gamma_hybrid_route import _states, periodic_grid_points

    try:
        rows = int(periodic_grid_points(problem))
        states = int(_states(problem))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return (rows, states) if rows * states < floor else None


def _opposite_spin_starts(problem) -> bool:
    """Whether the species of the atoms of ``problem`` start with polarizations of opposite sign (an antiferromagnetic
    or broken-symmetry start, ``Initial_Spin_Polarization``).  A function of the parsed input alone, the same on every
    rank and at every grid spacing."""

    species = getattr(problem, "pseudopotentials", None) or {}
    values = [
        float(getattr(species[label], "initial_spin_polarization", 0.0))
        for label in {atom.symbol for atom in problem.atoms}
        if label in species
    ]
    return any(value > 0.0 for value in values) and any(value < 0.0 for value in values)


def _cell_below_sector_floor(
    problem, mpi_context, *, spin_polarized: bool, floor: int, spin_floor: int, sector_forces: bool = False
) -> tuple[tuple[int, int] | None, int, str]:
    """``(rows and states below the floor or None, the floor, its switch)`` of a Gamma cell in the sectors.

    An unpolarized cell compares with ``floor`` (``PARSEC_PERIODIC_SECTOR_FLOOR``; a wire or slab under its ``auto``
    with :data:`WIRE_SLAB_FLOOR_ENTRIES`, :func:`_cell_sector_floor`, its sectors judged after the detection), a
    collinear spin-polarized one with
    a native LDA or GGA with ``spin_floor`` (``PARSEC_PERIODIC_SPIN_SECTOR_FLOOR``).  TS, LDA+U, a meta-GGA or libxc
    reach the sectors of a spin cell under their own switches and size rules and keep them.  So does an input whose
    species start with polarizations of opposite sign (:func:`_opposite_spin_starts`): its sectors and its full grid
    can end in different states, and a series in the grid spacing must stay on one route.
    """

    if not spin_polarized:
        floor = _cell_sector_floor(problem, floor)
        below = _periodic_basis_below_floor(problem, mpi_context, floor, sector_forces=sector_forces)
        return below, floor, _PERIODIC_FLOOR_SWITCH
    if _opposite_spin_starts(problem) or periodic_sector_refusal(problem, features=False) is not None:
        return None, spin_floor, _PERIODIC_SPIN_FLOOR_SWITCH
    below = _periodic_basis_below_floor(problem, mpi_context, spin_floor, sector_forces=sector_forces)
    return below, spin_floor, _PERIODIC_SPIN_FLOOR_SWITCH


def _orbital_group_refusal(problem, spin_polarized: bool, mpi_context, *, basis: int | None = None) -> str | None:
    """Why the orbitals keep the abelian group of before, or ``None`` where the weighted sectors serve.

    The sectors of a subgroup with weights (:class:`.Symmetry.PointGroupReduction`) solve one partner of a
    degenerate level where others exist: what reads every partner of every occupied state is refused here
    (exact exchange where :func:`_weighted_hybrid_refusal` says so, which is every hybrid under
    ``PARSEC_ORBITAL_GROUP_HYBRID=off``; the meta-GGA kinetic density only under ``PARSEC_ORBITAL_GROUP_META_GGA=off``,
    and that of a periodic cell under ``PARSEC_ORBITAL_GROUP_META_GGA_CELL=off`` too, whose ``auto`` values average
    ``tau`` and ``g`` over the full group since 2026-11-23; LDA+U only under ``PARSEC_LDAU_WEIGHTED_SECTORS=off``, whose
    ``auto`` averages its occupation matrices over the full group since 2026-11-21), as is spin-orbit coupling.  Since 2026-11-19 four
    inputs take the weighted sectors where their switch is ``auto`` (``Symmetry/weighted_inputs.py``):
    ``Fix_Number_Eigenvalues``, a libxc LDA or GGA, ``occup.in`` (equal occupations of partners, else the run
    starts again in the abelian sectors) and collinear spin in the symmetry sectors (``Spin_Symmetry_Sectors``;
    a channel pair whose guard fires starts again in the abelian sectors).  ``off`` refuses that input as before.
    A calculation prepared again after it left the weighted sectors keeps the abelian group
    (:func:`.Symmetry.weighted_inputs.abelian_orbitals`).  Since 2026-11-21 spin and ``occup.in`` take them on
    several MPI ranks too (``PARSEC_ORBITAL_GROUP_MPI=auto``), and a spin run whose ``basis`` (grid points times
    states of a channel; ``None`` where the grid is not built yet, which applies no floor) is below the floor of
    ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR`` keeps the abelian group (:func:`_weighted_spin_refusal`).
    """

    from parsec_python.V_xc.libxc import is_libxc_functional

    from .Symmetry.weighted_inputs import forced_abelian_reason, mpi_weighted_setting, weighted_input_setting

    scf = getattr(problem, "scf", None)
    if scf is None:
        return "no SCF settings"
    forced = forced_abelian_reason()
    if forced is not None:
        return forced
    if getattr(scf, "noncollinear", False) or getattr(scf, "self_consistent_spin_orbit", False):
        return "spin polarization or spin-orbit coupling"
    if spin_polarized:
        if weighted_input_setting("spin") == "off":
            return "spin polarization or spin-orbit coupling"
        spin_refusal = _weighted_spin_refusal(problem, mpi_context, basis=basis)
        if spin_refusal is not None:
            return spin_refusal
    if getattr(scf, "is_hybrid", False) or bool(getattr(mpi_context, "exchange", False)):
        # (the hybrid of a cluster in one process takes them under PARSEC_ORBITAL_GROUP_HYBRID=auto: its exchange
        # through the coset images of the weighted states, Hamiltonian/exchange_device.WeightedExchange)
        hybrid_refusal = _weighted_hybrid_refusal(problem, mpi_context)
        if hybrid_refusal is not None:
            return hybrid_refusal
    if getattr(scf, "is_meta_gga", False) and (
        weighted_input_setting("meta_gga") == "off"
        or (getattr(problem, "periodic_cell", None) is not None and weighted_input_setting("meta_gga_cell") == "off")
    ):
        # (a cluster's meta-GGA takes them under PARSEC_ORBITAL_GROUP_META_GGA=auto: tau and g of the solved states
        # averaged over the full group, Hamiltonian/meta_gga_device.py; a cell's, which the sectors take only as a
        # bulk cell at the Gamma point, under PARSEC_ORBITAL_GROUP_META_GGA_CELL=auto as well)
        return "a meta-GGA functional"
    if is_libxc_functional(getattr(scf, "xc_functional", "ca")) and weighted_input_setting("libxc") == "off":
        return "a libxc functional"
    if getattr(scf, "fixed_occupations", None) is not None:
        if weighted_input_setting("occupations") == "off":
            return "occupations from occup.in"
        if mpi_context is not None and mpi_weighted_setting() == "off":
            # (as before 2026-11-21: a file that fills partners unequally starts again in the abelian sectors on every
            # rank since, MPISectorContext.reprepare)
            return f"occupations from occup.in on several MPI ranks ({_MPI_WEIGHTED_OFF})"
    if any(getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values()):
        from .Hamiltonian.hubbard_device import LDAU_WEIGHTED_SECTORS_SWITCH, ldau_weighted_sectors_setting

        # (the occupation matrices of the weighted states averaged over the full group: HubbardGroupAverage)
        if ldau_weighted_sectors_setting() == "off":
            return f"LDA+U ({LDAU_WEIGHTED_SECTORS_SWITCH}=off)"
    if (
        getattr(getattr(problem, "eigensolver", None), "fixed_state_count", False)
        and weighted_input_setting("fixed_count") == "off"
    ):
        return "Fix_Number_Eigenvalues (the state counts of a restart file, sector by sector)"
    return None


_MPI_WEIGHTED_OFF = "PARSEC_ORBITAL_GROUP_MPI=off"


def _weighted_hybrid_refusal(problem, mpi_context) -> str | None:
    """Why a hybrid functional keeps the abelian group, or ``None`` where its exchange reads the weighted sectors.

    ``PARSEC_ORBITAL_GROUP_HYBRID=auto`` admits the self-consistent hybrid of a cluster in one process: the exchange of
    :class:`.Hamiltonian.exchange_device.WeightedExchange` (the coset images of the weighted states), in the plans
    and with the rules of every other input (spin and ``occup.in`` by their own switches above).  The exchange
    service of several MPI ranks (``SCF/hybrid_mpi.py``), a periodic cell's (the sectors of its little group,
    ``SCF/gamma_hybrid_route.py``), a meta-GGA hybrid and the units of ``PARSEC_ORBITAL_IRREPS`` keep the abelian
    group, as does every hybrid under ``off``, as before.
    """

    from .Symmetry.axis_reflection import orbital_irreps_setting
    from .Symmetry.weighted_inputs import weighted_input_setting

    scf = getattr(problem, "scf", None)
    if weighted_input_setting("hybrid") == "off":
        return "a hybrid functional"
    if mpi_context is not None or bool(getattr(mpi_context, "exchange", False)):
        return "a hybrid functional on several MPI ranks"
    if getattr(problem, "periodic_cell", None) is not None:
        return "a hybrid functional of a periodic cell"
    if getattr(scf, "is_meta_gga", False):
        return "a meta-GGA hybrid functional"
    if orbital_irreps_setting() != "off":
        return "a hybrid functional with the units of PARSEC_ORBITAL_IRREPS"
    return None


def _weighted_spin_refusal(problem, mpi_context, *, basis: int | None = None) -> str | None:
    """Why a collinear spin-polarized input keeps the abelian group with ``PARSEC_ORBITAL_GROUP_SPIN=auto``.

    The weighted sectors serve two channels that the symmetry sectors solve: a cluster whose
    ``Spin_Symmetry_Sectors`` takes them (:func:`_spin_sector_refusal`), or a bulk cell at the Gamma point on the
    route of its sectors (``SCF/gamma_spin.py``, which :func:`periodic_sector_refusal` has admitted), in one process
    and, since 2026-11-21, on several MPI ranks (``PARSEC_ORBITAL_GROUP_MPI=auto``: every rank prepares the same plan,
    and a run that leaves them is prepared again in the abelian sectors on every rank; ``off`` keeps the abelian
    group there, as before).  Two channels on the full grid read no group of the orbitals; they keep the detection
    of before.  A ``basis`` (grid points times states of a channel) below ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR`` keeps
    the abelian group, whose sectors are as fast there (``None``: not known yet, no floor).
    """

    from .Symmetry.weighted_inputs import SPIN_FLOOR_SWITCH, mpi_weighted_setting, spin_floor_setting

    if mpi_context is not None and mpi_weighted_setting() == "off":
        return f"spin polarization on several MPI ranks ({_MPI_WEIGHTED_OFF})"
    if _spin_sector_refusal(problem, mpi_context=mpi_context) is not None:
        return "spin polarization on the full grid"
    floor = spin_floor_setting()
    if basis is not None and int(basis) < floor:
        return (
            f"spin polarization with {int(basis)} grid points x states, below the floor of the weighted spin sectors "
            f"({floor}, {SPIN_FLOOR_SWITCH})"
        )
    return None


def _spin_floor_basis(problem, grid) -> int | None:
    """Grid points of ``grid`` times the states of a channel of ``problem`` (``States_Num``, or half its valence
    electrons and six): what ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR`` compares, the same on the thread of the maps and in
    line (:func:`_weighted_spin_refusal`); ``None`` where it cannot be counted (no floor then)."""

    from .SCF.gamma_hybrid_route import _states

    try:
        return int(grid.size) * int(_states(problem))
    except (AttributeError, OSError, TypeError, ValueError, KeyError):
        return None


def _kept_abelian_note(refusal) -> str | None:
    """The NOTE of a run whose spin or ``occup.in`` keeps the abelian group by a rule of 2026-11-21 (``None`` for any
    other ``refusal`` of :func:`_orbital_group_refusal`): ``PARSEC_ORBITAL_GROUP_MPI=off`` on several ranks, or the
    floor of the weighted spin sectors.  The larger point group is then not looked for."""

    from .Symmetry.weighted_inputs import SPIN_FLOOR_SWITCH

    if refusal is None or not (_MPI_WEIGHTED_OFF in refusal or SPIN_FLOOR_SWITCH in refusal):
        return None
    return (
        f"the orbitals keep the sectors of the abelian group ({refusal}); the weighted sectors of a larger point "
        "group are not looked for"
    )


def _sectors_keep_orbits(reference, setting: str, reduction, atoms, *, grid=None) -> bool:
    """Whether the sectors of ``reduction`` take the projectors of one atom per orbit.  Without LDA+U: where the atoms
    keep the group of the sectors within 1e-6 bohr (:func:`.Symmetry.axis_reflection.sector_orbits_exact`,
    ``PARSEC_SECTOR_ORBITS_EXACT``).  With LDA+U: under ``PARSEC_LDAU_PROJECTOR_ORBITS=auto``, by the same distance
    (``hubbard_device.ldau_orbits_exact``).  A structure accepted within a looser ``PARSEC_SYMMETRY_ATOM_TOLERANCE``
    keeps the projectors of every atom.  ``grid``: the grid of the detection where it is not that of ``reference`` (the
    frame of the grid indices of an oblique cell, ``Symmetry/oblique_frame.py``, with ``atoms`` in that frame)."""

    if grid is None:
        grid = reference.grid
    if getattr(reference, "hubbard", None) is None:
        from .Symmetry.axis_reflection import sector_orbits_exact

        return sector_orbits_exact(reduction, atoms, grid)
    if setting != "auto":
        return False
    from .Hamiltonian.hubbard_device import ldau_orbits_exact

    return ldau_orbits_exact(reduction, atoms, grid)


#: The report of an LDA+U run whose atoms keep the group of its sectors only within a loose atom tolerance.
_LDAU_LOOSE_ORBITS_NOTE = (
    "LDA+U keeps the projectors of every atom: the atoms keep the group only to more than 1e-6 bohr "
    "(PARSEC_SYMMETRY_ATOM_TOLERANCE)"
)

#: The same report of a run without LDA+U (``PARSEC_SECTOR_ORBITS_EXACT``).
_SECTOR_LOOSE_ORBITS_NOTE = (
    "the sectors keep the projectors of every atom: the atoms keep the group only to more than 1e-6 bohr "
    "(PARSEC_SYMMETRY_ATOM_TOLERANCE, PARSEC_SECTOR_ORBITS_EXACT)"
)


def _without_atom_orbits(reduction):
    """``reduction`` without the orbits of its atoms (the projectors of every atom in its sectors).

    LDA+U appends fifteen columns per LDA+U atom to the KB factor, whose coefficients every SCF iteration writes
    atom by atom (``Hamiltonian/hubbard_device.py``); under ``PARSEC_LDAU_PROJECTOR_ORBITS=off`` the sectors keep
    the columns of every atom, as before 2026-11-21 (``auto``, the default, keeps the orbits: the coefficients of
    the representatives, ``hubbard_device.HubbardOrbits``; not for atoms off the group by more than 1e-6 bohr,
    :func:`_sectors_keep_orbits`, which applies that distance to the KB columns of a run without LDA+U too).  The
    reductions
    of a cell with translations (``SCF/kpoint_sectors.LittleGroupReduction``) know their atom orbits from the
    start (one atom per orbit, :func:`.Symmetry.axis_reflection.compress_projectors_for_sectors`): a copy without
    them.  ``reduction`` itself where it knows none.
    """

    import dataclasses

    if all(getattr(reduction, name, None) is None for name in ("atom_representatives", "atom_orbit_sizes")):
        return reduction
    names = {field.name for field in dataclasses.fields(reduction)} if dataclasses.is_dataclass(reduction) else set()
    if not {"atom_representatives", "atom_orbit_sizes"} <= names:
        raise RuntimeError(f"the atom orbits of a {type(reduction).__name__} cannot be dropped for LDA+U")
    return dataclasses.replace(reduction, atom_representatives=None, atom_orbit_sizes=None)


def _sector_count(reduction) -> int:
    """The number of orbital sectors of ``reduction``: its order for an abelian group, the solved one-dimensional
    sectors of the subgroup for the weighted sectors of a point group (fewer than its order)."""

    from .Symmetry import PointGroupReduction

    if isinstance(reduction, PointGroupReduction):
        return int(np.asarray(reduction.sector_characters).shape[0])
    return int(reduction.group_order)


def _ranks_divide_sectors(reduction, mpi_context) -> bool:
    """Whether the ranks of an MPI run can take the sectors of ``reduction`` (``True`` in one process).

    The sector context (:meth:`.experimental.mpi_scf.MPISectorContext.divides`) gives one rank with the GPUs of its
    node at least one sector: the weighted sectors of D2d (three) on four such ranks, or the two of a cell repeated
    twice, would stop the run in its sector configuration.  A group that a rank count cannot divide is not taken;
    the orbitals keep the group of before (the abelian one about the origin), which ran there.  The placement of
    the sectors by size (``PARSEC_SECTOR_PLACEMENT=size``, the default) asks the same: every rank of the GPUs of
    its node still owns one group of sectors or more, and ranks of one GPU each take any count of sectors (a group
    of them may solve several), so the counts that divide are those of the placement by count.
    """

    divides = getattr(mpi_context, "divides", None) if mpi_context is not None else None
    if not callable(divides):
        return True
    return bool(divides(_sector_count(reduction)))


def _native_extension_reason(selection) -> str:
    """Why the native C++ extension serves no component of a CuPy run: ``--backend cupy``, or no extension."""

    if selection.requested == "cupy":
        return "the CuPy backend was named (--backend cupy), which leaves the native C++ extension unused"
    reasons = "; ".join(str(reason) for reason in getattr(selection, "fallback_reasons", ()))
    return "the native C++ extension is not available" + (f" ({reasons})" if reasons else "")


def _cluster_full_grid_notes(selection, *, periodic: bool, group_found: bool, orbitals_in_sectors: bool) -> list[str]:
    """The NOTE of a cluster with a point group whose Hartree solve takes the full grid for want of the extension.

    The symmetry-reduced Hartree solve of a cluster (the CG on the wedge and
    its boundary values) is a route of the native extension.  With the CuPy
    backend alone, an explicit ``--backend cupy`` or ``auto`` where the
    extension is not built, the Poisson CG runs on the full grid with the
    full-grid finite-difference matrix on the device, while the orbitals stay
    in the sectors: more memory and time for a large cluster.  Nothing of the
    run changes here; before 2026-11-18 only the ``hartree_backend`` line of
    the report showed it.  Empty for a periodic cell (its Hartree solve has
    a route of its own), a cluster without a group and the native Hartree
    backend.
    """

    if periodic or not group_found or selection.selected != "cupy" or selection.hartree_backend == "native":
        return []
    return [
        "the Hartree solve of this cluster runs on the full grid (a CuPy CG with the full-grid finite-difference "
        "matrix on the device), not on the wedge of its point group, because "
        + _native_extension_reason(selection)
        + ("; the orbitals stay in the symmetry sectors" if orbitals_in_sectors else "")
        + ". The default backend (auto) with the native extension solves it on the wedge."
    ]


def _cluster_xc_full_grid_note(selection, *, functional: str) -> str:
    """The NOTE of a cluster in the sectors whose CA-LDA is evaluated on the full grid of the host.

    The CA-LDA evaluator on the orbits (``NativeCALDAEvaluator``) is a kernel
    of the native extension; without it the orbit density is expanded to the
    full grid every step, evaluated by the reference functional on the host
    and reduced again (an exact adapter, the same numbers).
    """

    return (
        f"the XC functional ({functional}) of this cluster is evaluated on the full grid of the host every SCF "
        "step (the orbit fields expanded and reduced again), because "
        + _native_extension_reason(selection)
        + ". The default backend (auto) with the native extension evaluates it on the orbits."
    )


def _orbital_reduction(symmetry_reduction, stabilizer_disabled: bool):
    """Return the reduction the orbitals take of a detected one, or ``None``.

    With the stabilizer policy off, a group that fixes grid points (orbits
    of unequal size) is left to the Hartree solver alone and the orbitals
    use the full grid: an A/B control and a conservative override.
    """

    if (
        symmetry_reduction is not None
        and not np.all(
            symmetry_reduction.multiplicities == symmetry_reduction.group_order
        )
        and stabilizer_disabled
    ):
        return None
    return symmetry_reduction


def sector_palette_ladder_setting() -> str:
    """``PARSEC_SECTOR_PALETTE_LADDER``: ``auto`` (default) or ``off``.

    The sector stencils of a periodic cell code their coefficients in one byte (256 values) where they have at most
    256, in two (65536) where they have more (``PARSEC_SECTOR_PALETTE_CODES=auto``, the default since 2026-11-19), or
    in one byte alone (``PARSEC_SECTOR_PALETTE_CODES=off``; :func:`_sector_palette_check`).  Where those of the group
    detected need more than their codes hold (under ``off``: an even repeat of a cubic cell on ``n = 2 mod 4`` points
    per edge, a cell of unequal steps at a high order), ``auto`` steps down to a group whose
    stencils fit, the same on every MPI rank: the weighted sectors of a subgroup of the point group, then the
    commuting involutions with their translations (:func:`.Symmetry.axis_reflection.periodic_abelian_reduction`), then
    the commuting signed permutations about the origin (D2 for a cubic crystal), then the full grid (the identity
    sector of an MPI run).  ``off`` goes to the full grid at once, as a cell whose grid steps differ between the axes
    did before 2026-11-19; a cube was not checked then and stopped in the build of its stencils.
    """

    value = os.environ.get("PARSEC_SECTOR_PALETTE_LADDER", "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"PARSEC_SECTOR_PALETTE_LADDER must be auto or off, not {value!r}")
    return value


def _partner_rows_note(candidate) -> str:
    """``, n of them on the orbits of larger subgroups (PARSEC_ORBITAL_IRREP_ROWS)`` where units of ``candidate`` are
    solved on partner rows, else empty."""

    from .Symmetry.axis_reflection import orbital_irrep_rows_setting

    cosets = getattr(candidate, "sector_partner_cosets", None)
    if cosets is None or orbital_irrep_rows_setting() != "partner":
        return ""
    projected = getattr(candidate, "sector_projected", None)
    count = sum(
        1 for index, item in enumerate(cosets) if item is not None and (projected is None or bool(projected[index]))
    )
    return f", {count} of them on the orbits of larger subgroups (PARSEC_ORBITAL_IRREP_ROWS)" if count else ""


def _symmetry_detection_text(candidate) -> str:
    """The ``symmetry_detection`` of the backend details for the group the orbitals of a run take."""

    from .Symmetry import PointGroupReduction, SignedPermutationReduction
    from .Symmetry.axis_reflection import holds_general_operations

    # (the frame of an oblique cell under PARSEC_OBLIQUE_FULL_GROUP=auto: integer maps of its grid indices, some of
    # them no signed permutation)
    kind = "integer maps of the grid indices" if holds_general_operations(candidate) else "signed permutations"
    return (
        (
            f"exact group of {candidate.full_operations.shape[0]} operations detected ({kind}, "
            f"{int(np.count_nonzero(np.any(np.asarray(candidate.full_translations) != 0, axis=1)))}"
            f" of them with a translation, by half the cell or, non-symmorphic, a fraction of it, or "
            f"about a symmetric site off the origin; PARSEC_PERIODIC_TRANSLATIONS, "
            f"PARSEC_PERIODIC_SYMMETRY_CENTRE)"
            if getattr(candidate, "full_translations", None) is not None
            else f"exact point group of {candidate.full_operations.shape[0]} {kind} detected"
        )
        + f"; the orbitals use the one-dimensional sectors of a subgroup of {candidate.group_order} "
        f"(weights {', '.join(str(int(w)) for w in candidate.sector_weights)})"
        + (
            f", as {len(candidate.sector_weights)} units of the irreducible representations of the point group, "
            f"{int(np.count_nonzero(candidate.sector_projected))} of them projected (PARSEC_ORBITAL_IRREPS)"
            + _partner_rows_note(candidate)
            if getattr(candidate, "sector_projected", None) is not None
            else ""
        )
        if isinstance(candidate, PointGroupReduction)
        else (
            f"exact commuting group of {candidate.group_order} operations detected (signed "
            f"permutations, "
            f"{int(np.count_nonzero(np.any(np.asarray(candidate.translations) != 0, axis=1)))} of them "
            f"with a translation; PARSEC_PERIODIC_ABELIAN_TRANSLATIONS)"
        )
        if getattr(candidate, "translations", None) is not None
        else "exact commuting signed-permutation subgroup detected"
        if isinstance(candidate, SignedPermutationReduction)
        else "exact Cartesian axis-reflection subgroup detected"
    )


def _below_minimum_sector_order(reduction, minimum: int) -> bool:
    """Whether the sectors of ``reduction`` have fewer operations than ``minimum`` (``minimum_sector_order`` of
    :func:`prepare_single_point`; ``0`` or ``1`` keeps every group)."""

    return int(minimum) > 1 and reduction is not None and 1 < int(getattr(reduction, "group_order", 1)) < int(minimum)


def _minimum_sector_order_text(reduction, minimum: int, found: str) -> str:
    """The ``symmetry_detection`` of a cell whose sectors :func:`_below_minimum_sector_order` dropped."""

    return (
        f"skipped: the sectors of {int(reduction.group_order)} operations found for the cell are fewer than the "
        f"{int(minimum)} from which PARSEC_MOVING_SECTORS=auto takes them for molecular dynamics, manual movements and "
        f"the relaxations with fixed atoms of the Fortran-derived routines (on takes them); found: {found}; the full "
        "grid is used"
    )


def _palette_group_label(reduction) -> str:
    """What the NOTE of the palette ladder calls the sectors of ``reduction``."""

    from .Symmetry import PointGroupReduction

    if isinstance(reduction, PointGroupReduction):
        return f"the weighted sectors of a subgroup of {int(reduction.group_order)} operations of the point group"
    translations = getattr(reduction, "translations", None)
    if translations is not None and np.any(np.asarray(translations) != 0):
        return f"the sectors of the {int(reduction.group_order)} commuting operations with their translations"
    return f"the sectors of the {int(reduction.group_order)} commuting signed permutations about the origin"


def _sector_palette_check(grid, reduction, build_decomposition, count=None, bound=None, *, label=None):
    """``(checked, refusal)`` of the sector stencils of a periodic cell.

    A packed stencil codes its coefficients in one byte (``PALETTE_CAPACITY``, 256 values), a sector stencil of more
    in two (65536, ``PARSEC_SECTOR_PALETTE_CODES=auto``; ``off``: one byte alone, a capacity of 256).  Those of a sector are its
    stencil values (``1 + M`` on a cube, up to ``1 + 3M`` where the grid steps differ between the axes) times the
    phases ``+-1`` and the multiplicity ratios ``sqrt(m_row / m_column)`` of its orbits, and the sums of the rows
    whose stencil holds two points of one orbit.  A cube whose orbits differ in size (an even repeat of a smaller
    cell on ``n = 2 mod 4`` points per edge: Si 2x2x2 at Grid_Spacing 0.6, 34^3 points, 295 values at the default
    order 12), a tetragonal 2x2x2 perovskite at the default order or Si at order 18 need more than 256: their sector
    stencils take two-byte codes, or under ``PARSEC_SECTOR_PALETTE_CODES=off`` the build of the stencils would stop
    the run.  Checked here on every grid, the same on every rank: first a bound
    (``bound``, by default ``sector_palette_bounds``: the whole table of values and the exact sums of the rows near
    a symmetry element, 1/|H| of the grid and a few percent of its rows; nothing more for the 18 points per edge of
    the diamond benchmarks), then the count of the builder itself (``count``, by default ``sector_palette_sizes``)
    for the sectors whose bound exceeds 256 values.  ``refusal`` says why the stencils of ``reduction`` (``label``)
    do not fit; ``checked`` is ``(orbital reduction, decomposition, largest count)`` where they do, for the stage of
    the orbitals (the largest count is 0 where the bound proves 256 values or fewer, and above 256 where two-byte
    codes hold them).
    ``build_decomposition(orbital reduction)`` gives what that stage would; nothing is checked where it fails (the
    stage of the orbitals goes on as before).
    """

    if reduction is None or reduction.group_order <= 1:
        return None, None
    try:
        orbital = _orbital_reduction(reduction, _stabilizer_symmetry_disabled())
    except ValueError:
        # (refused where the stage of the orbitals reads it)
        return None, None
    if orbital is None or orbital.group_order <= 1:
        return None, None
    from .Symmetry.sector_stencil import (
        PALETTE_CAPACITY,
        SectorStencilUnavailable,
        sector_palette_bounds,
        sector_palette_capacity,
        sector_palette_sizes,
    )

    capacity = sector_palette_capacity()
    try:
        built = build_decomposition(orbital)
        bounds = (sector_palette_bounds if bound is None else bound)(grid, built[0], wrap=True)
        # (counted above 256 under every setting: the count says whether two-byte codes are needed)
        over = [index for index, size in enumerate(bounds) if size is not None and size > PALETTE_CAPACITY]
        sizes = (sector_palette_sizes if count is None else count)(grid, built[0], over, wrap=True) if over else ()
    except (SectorStencilUnavailable, ValueError, RuntimeError):
        return None, None
    largest = max((size for size in sizes if size is not None), default=0)
    if largest <= capacity:
        return (orbital, built, largest), None
    return None, (
        f"the sector stencils of {label or _palette_group_label(orbital)} need {largest} distinct coefficients, more "
        f"than the {capacity} of their palette (Expansion_Order {int(grid.settings.expansion_order)}"
        + (", PARSEC_SECTOR_PALETTE_CODES=off" if capacity == PALETTE_CAPACITY else "")
        + ")"
    )


def _readable_xc_label(xc_functional: str, *, spin_polarized: bool) -> str:
    """Backend label of a functional the readable ``V_xc`` package evaluates."""

    from parsec_python.V_xc.selection import XC_DISPLAY_NAMES

    name = XC_DISPLAY_NAMES.get(xc_functional, xc_functional)
    if xc_functional == "pw91":
        name = f"discrete-variational {name}"
    if str(xc_functional).startswith("libxc:"):
        # libxc's C code on the host (a host build, or PARSEC_CUPY_XC=off).
        return f"host libxc (C) {'spin-polarized ' if spin_polarized else ''}{name}"
    return (
        f"vectorized NumPy {'spin-polarized ' if spin_polarized else ''}{name}"
    )


def _readable_spin_xc_label(xc_functional: str) -> str:
    """Backend label of a spin functional the readable ``V_xc`` package evaluates."""

    if xc_functional == "pbe":
        return "vectorized NumPy discrete-variational spin-polarized PBE"
    if xc_functional == "ca":
        return "vectorized NumPy CA/PZ local spin density"
    return _readable_xc_label(xc_functional, spin_polarized=True)


def _spin_sector_refusal(problem, *, mpi_context=None) -> str | None:
    """Why a spin-polarized ``problem`` keeps both channels on the full grid, or ``None``.

    ``Spin_Symmetry_Sectors`` (``SCFSettings.spin_symmetry_sector_mode``):
    ``off`` keeps the full grid; ``auto`` takes the symmetry sectors of the
    point group where the run can use them (the driver still needs a group,
    the CuPy backend and symmetry on), except for what the spin loop of the
    sectors does not carry: LDA+U of a cluster under ``PARSEC_LDAU_SPIN_SECTORS``
    ``auto`` and of every input under ``off`` (the default)
    (``SCF/gamma_features.py``; where it takes them, each channel writes its own
    potential matrices into the shared sector operators before its solve), the
    triangular Ritz hand-over and, in one process, a filter spread over several devices
    (``PARSEC_CUPY_DISTRIBUTED_FILTER=1``).  On the ranks of the MPI runner (``mpi_context``) that filter keeps the
    sectors since 2026-11-22: it spreads the filter of a sector over the GPUs of its group and reads the potential of
    the sector operator at every application, which each channel writes before its solve; the runner had no full
    grid to fall back to and refused every spin-polarized run under that value (the launch settings of
    ``scripts/run_multi_gpu_scf.sh``).
    ``Symmetrize_Charge_Density`` averages each
    channel on the orbits of the sectors under
    ``PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto`` (since 2026-11-19,
    :mod:`.SCF.density_symmetry`); ``off`` keeps the full grid with it, as
    before.
    """

    from .SCF.density_symmetry import SYMMETRIZED_DENSITY_SECTORS_SWITCH, symmetrized_density_sectors_setting
    from .SCF.gamma_features import LDAU_SPIN_SWITCH, ldau_spin_sector_setting

    scf = problem.scf
    mode = scf.spin_symmetry_sector_mode
    if mode == "off":
        return "Spin_Symmetry_Sectors: off"
    species = getattr(problem, "pseudopotentials", {}).values()
    for condition, reason in (
        (
            any(getattr(item, "has_hubbard", False) for item in species)
            and (
                ldau_spin_sector_setting() == "off"
                or (ldau_spin_sector_setting() == "auto" and getattr(problem, "periodic_cell", None) is None)
            ),
            f"LDA+U ({LDAU_SPIN_SWITCH}={ldau_spin_sector_setting()})",
        ),
        (
            bool(getattr(scf, "symmetrize_density", False)) and symmetrized_density_sectors_setting() == "off",
            f"Symmetrize_Charge_Density ({SYMMETRIZED_DENSITY_SECTORS_SWITCH}=off)",
        ),
        (
            os.environ.get("PARSEC_CUPY_RITZ_HANDOVER", "").strip().lower() == "triangular",
            "PARSEC_CUPY_RITZ_HANDOVER=triangular",
        ),
        (
            mpi_context is None and os.environ.get("PARSEC_CUPY_DISTRIBUTED_FILTER", "").strip().lower() == "1",
            "PARSEC_CUPY_DISTRIBUTED_FILTER=1",
        ),
    ):
        if condition:
            return f"Spin_Symmetry_Sectors: auto keeps the full grid with {reason}"
    return None


#: Appended to the symbol of an atom that Move_Flag fixes, for the symmetry detection alone (:func:`_detection_atoms`).
FIXED_ATOM_LABEL = "|fixed"


def _detection_atoms(atoms, fixed_atoms):
    """The atoms the symmetry detection reads: ``atoms`` itself, or with the fixed ones as a species of their own.

    ``fixed_atoms`` (``True`` for an atom that ``Move_Flag`` holds, :func:`parsec_python.Relaxation.run.
    symmetry_fixed_atoms`) makes every operation the detection finds map the fixed atoms onto fixed atoms of their
    species, the subgroup a relaxation with those flags keeps (``PARSEC_MOVE_FLAG_SUBGROUP=auto``).  The labels are
    read by the detection only: the operations found are symmetries of the atoms themselves, which every later stage
    reads.  ``None`` returns ``atoms`` unchanged (the same object).
    """

    if fixed_atoms is None:
        return atoms
    atoms = tuple(atoms)
    if len(fixed_atoms) != len(atoms):
        raise ValueError(
            f"one Move_Flag per atom is required for the symmetry detection: {len(fixed_atoms)} flags, "
            f"{len(atoms)} atoms"
        )
    return tuple(
        replace(atom, symbol=atom.symbol + FIXED_ATOM_LABEL) if fixed else atom
        for atom, fixed in zip(atoms, fixed_atoms)
    )


def _spin_flip_route(problem, reference, *, reducer, decomposition):
    """``(operation, reason)`` of ``PARSEC_SPIN_FLIP_SYMMETRY=auto`` for a prepared spin-polarized run.

    The operation of :func:`.SCF.spin_flip.detect_spin_flip` on the grid and atoms of ``reference``, with the
    orbits of the sector fields (``reducer``) and the sectors (``decomposition``) where the run has them, the
    start densities of both channels and the local ionic potential (point charges included); ``None`` and why for
    what the route does not take (a hybrid, a meta-GGA, LDA+U, ``Symmetrize_Charge_Density``, spin-orbit coupling)
    and for a run without such an operation.  The spin loop checks the operation again for every loop it runs
    (the atoms, the ionic potential with the field of a ``Polarizability`` loop, the potentials of a warm start).
    """

    from .SCF.spin_flip import detect_spin_flip

    scf = problem.scf
    species = dict(getattr(problem, "pseudopotentials", {}))
    for condition, reason in (
        (bool(getattr(scf, "is_hybrid", False)), "a hybrid functional"),
        (bool(getattr(scf, "is_meta_gga", False)), "a meta-GGA"),
        (any(getattr(item, "has_hubbard", False) for item in species.values()), "LDA+U"),
        (bool(getattr(scf, "symmetrize_density", False)), "Symmetrize_Charge_Density"),
        (
            bool(getattr(scf, "noncollinear", False) or getattr(scf, "self_consistent_spin_orbit", False))
            or any(getattr(item, "spin_orbit", False) for item in species.values()),
            "spin-orbit coupling or non-collinear magnetism",
        ),
    ):
        if condition:
            return None, reason
    densities = getattr(reference, "initial_spin_densities", None)
    if densities is None:
        return None, "the run has no start densities of the two channels"
    fixed = getattr(scf, "fixed_occupations", None)
    try:
        return detect_spin_flip(
            reference.grid,
            reference.atoms,
            species,
            electron_count=float(reference.electron_count),
            start_densities=densities,
            field_reduction=None if reducer is None else reducer.reduction,
            decomposition=decomposition,
            fixed_occupations=None if fixed is None else (fixed, getattr(scf, "fixed_occupations_down", None)),
            ionic_potential=getattr(reference, "ionic_potential", None),
        )
    except ValueError as error:
        return None, f"{error}"


def _same_atoms(left, right) -> bool:
    """Whether two atom sequences name the same species at the same positions, bit for bit."""

    if len(left) != len(right):
        return False
    if any(a.symbol != b.symbol for a, b in zip(left, right)):
        return False
    return np.array_equal(
        np.asarray([atom.position for atom in left], dtype=np.float64),
        np.asarray([atom.position for atom in right], dtype=np.float64),
    )


class _OverlappedSymmetryMaps:
    """The symmetry maps of a rank, built on a thread beside its projectors.

    Symmetry detection and the representation phases read the grid and the
    atoms and nothing else of the reference system, and the projectors do
    not read them.  The thread starts when the grid exists and calls the
    two functions the calling thread would call after the reference
    preparation, with the same arguments: the maps are theirs.  It touches
    no device and makes no MPI call, so its end leaves an open graph
    capture valid (``backends/cupy_capture.py``).

    The calling thread takes the reduction and the decomposition where it
    used to build them, and an error of either is raised there, by the
    name and in the order it had in line.  Measured on the host of an A100
    node with 16 OpenMP threads, the projectors beside the maps took 1.02 to
    1.08 times the longer of the two (3,480 to 39,368 electrons).

    The thread publishes the reduction when it is detected, for a stage of
    another thread that can use it before the calling thread has joined.
    """

    def __init__(self, *, cache_directory, reduction_known, stages) -> None:
        self._cache_directory = cache_directory
        self._published = reduction_known
        self._stages = stages
        self._executor = None
        self._grid = None
        self._atoms = None
        self._reduction: Future = Future()
        self._decomposition: Future = Future()
        # The weighted sectors of a larger subgroup (``detect_point_group_reduction``), where the caller
        # asks for them (:meth:`start`): detected here after the reduction, and decomposed in its place.
        self._point_group: Future = Future()
        # Set by the caller where the weighted sectors serve the orbitals (no refusal of
        # ``_orbital_group_refusal``): :meth:`start` then detects them on this thread.  ``point_group_wanted_for``
        # (a callable of the grid, or None) decides it on the grid :meth:`start` is given where the refusal depends
        # on its size (the floor of the weighted spin sectors, PARSEC_ORBITAL_GROUP_SPIN_FLOOR).
        self.point_group_wanted = False
        self.point_group_wanted_for = None
        # Whether the MPI ranks can take the sectors of a reduction (_ranks_divide_sectors; None: one process).
        # Where they cannot take the weighted ones, the caller keeps the reduction of before, which is decomposed.
        self.ranks_divide = None
        # The atoms Move_Flag fixes, a species of their own for the detection (_detection_atoms; None: none).
        self.fixed_atoms = None
        # PARSEC_IONIC_SUM_GROUP (set by the caller): ``full`` publishes the orbits of the full point group found
        # after the reduction, ``first`` the reduction as soon as it is detected.
        self.ionic_sum_group = "first"
        self.seconds = 0.0
        self.wait_seconds = 0.0

    @property
    def started(self) -> bool:
        return self._executor is not None

    def start(self, grid, atoms, *, decompose: bool, point_group: bool = False) -> None:
        """Begin the maps of ``grid`` and ``atoms``.

        With ``decompose`` the representation phases follow the detection
        where the calling thread would build them: for a group with an
        operation beside the identity that the orbitals take.  With
        ``point_group`` the larger group of the weighted sectors is detected
        after the reduction (``detect_point_group_reduction``, which returns
        the reduction itself where it gains nothing), and the phases are
        those of what it returns: the calling thread would detect it and
        decompose it in line, after the reference preparation.
        """

        if self._executor is not None:
            raise RuntimeError("the symmetry maps of a preparation start once")
        wanted = self.point_group_wanted
        if wanted and callable(self.point_group_wanted_for):
            wanted = bool(self.point_group_wanted_for(grid))
        point_group = bool(point_group or wanted)
        # Imported by the calling thread, as it imports them in line.
        from .Symmetry import (
            load_or_build_reflection_decomposition,
            load_or_detect_reflection_reduction,
        )
        from .Symmetry.axis_reflection import symmetry_atom_tolerance

        # (the tolerance of the detection in line, PARSEC_SYMMETRY_ATOM_TOLERANCE)
        atom_tolerance = symmetry_atom_tolerance()
        # (and its atoms: the fixed ones of a relaxation as a species of their own, PARSEC_MOVE_FLAG_SUBGROUP)
        atoms = _detection_atoms(atoms, self.fixed_atoms)
        self._grid, self._atoms = grid, atoms
        stages, cache_directory = self._stages, self._cache_directory

        def build():
            started = perf_counter()
            try:
                try:
                    found = load_or_detect_reflection_reduction(
                        grid, atoms, cache_directory=cache_directory, atom_tolerance=atom_tolerance
                    )
                except BaseException as error:
                    self._reduction.set_exception(error)
                    return
                finally:
                    stages.add("symmetry_reduction", started)
                self._reduction.set_result(found)
                candidate, cache_info = found
                # PARSEC_IONIC_SUM_GROUP=full: the ionic sums wait for the point group below and sum on the orbits
                # of its full group; ``first`` publishes this reduction at once, as before.
                wait_for_group = (
                    point_group and candidate.group_order > 1 and self.ionic_sum_group == "full"
                )
                if not wait_for_group:
                    self._published.publish(candidate)
                if point_group and candidate.group_order > 1:
                    from .Symmetry import detect_point_group_reduction

                    mark = perf_counter()
                    try:
                        candidate = detect_point_group_reduction(
                            grid, atoms, candidate, atom_tolerance=atom_tolerance
                        )
                    except BaseException as error:
                        self._published.publish(found[0])
                        self._point_group.set_exception(error)
                        return
                    finally:
                        stages.add("orbital_group", mark)
                    if wait_for_group:
                        # (the orbits of the full group, then those of H and of the abelian group found first)
                        self._published.publish(_ionic_sum_candidates(candidate, stages, first=found[0]))
                    self._point_group.set_result(candidate)
                    if (
                        candidate is not found[0]
                        and self.ranks_divide is not None
                        and not self.ranks_divide(candidate)
                    ):
                        candidate = found[0]
                if not decompose or candidate.group_order <= 1:
                    return
                try:
                    disabled = _stabilizer_symmetry_disabled()
                except ValueError:
                    # The calling thread refuses the value where it reads it.
                    return
                if _orbital_reduction(candidate, disabled) is None:
                    return
                mark = perf_counter()
                try:
                    built = load_or_build_reflection_decomposition(
                        grid,
                        candidate,
                        reduction_key=cache_info.key,
                        cache_directory=cache_directory,
                    )
                except BaseException as error:
                    self._decomposition.set_exception(error)
                else:
                    self._decomposition.set_result((candidate, built))
                finally:
                    stages.add("symmetry_decomposition", mark)
            finally:
                self.seconds = perf_counter() - started
                # Whatever happened, nobody is left waiting.
                self._published.publish(None)
                for future in (self._reduction, self._decomposition, self._point_group):
                    if not future.done():
                        future.set_result(None)

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-symmetry-setup",
        )
        self._executor.submit(build)
        # The thread ends with its task, joined or not.
        self._executor.shutdown(wait=False)

    def serves(self, grid, atoms) -> bool:
        """Whether the thread was started on this very grid and these atoms."""

        return (
            self._executor is not None
            and grid is self._grid
            and _same_atoms(atoms, self._atoms)
        )

    def reduction(self):
        """Wait for the detection; return what it returned or raise what it raised."""

        started = perf_counter()
        try:
            return self._reduction.result()
        finally:
            self.wait_seconds += perf_counter() - started
            self._stages.add("symmetry_reduction_wait", started)

    def point_group(self):
        """Wait for the detection of the weighted sectors; ``None`` where the thread was not asked for it."""

        started = perf_counter()
        try:
            return self._point_group.result()
        finally:
            self.wait_seconds += perf_counter() - started
            self._stages.add("orbital_group_wait", started)

    def decomposition(self, reduction):
        """Join the thread; return the decomposition it built of ``reduction``.

        ``None`` where it built none, or one of another reduction; an error
        of the build is raised.
        """

        started = perf_counter()
        try:
            built = self._decomposition.result()
        finally:
            self._executor.shutdown(wait=True)
            self.wait_seconds += perf_counter() - started
            self._stages.add("symmetry_decomposition_wait", started)
        if built is None or built[0] is not reduction:
            return None
        return built[1]

    def close(self) -> None:
        """Wait for the thread without reading what it built or raised."""

        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def details(self) -> tuple[tuple[str, str], ...]:
        """Report where the maps were built, and for the thread its time and the waits at its joins."""

        if self._executor is None:
            return (("symmetry_setup", "inline"),)
        return (
            ("symmetry_setup", "overlapped with the projectors"),
            ("symmetry_setup_seconds", f"{self.seconds:.6f}"),
            ("symmetry_setup_wait_seconds", f"{self.wait_seconds:.6f}"),
        )


class _Preparation:
    """What the stages of one preparation share.

    ``stages`` is its clock.  A stage that starts a thread says here how to
    join it, and :meth:`close` joins whatever still runs: the preparation
    leaves no thread behind, whether it returns or raises.

    ``symmetry_reduction`` is the reduction the preparation has detected,
    published once it is known: ``None`` where the detection failed and
    where the preparation looks for none, and possibly a group of the
    identity alone.  ``symmetry_maps`` is the thread that builds the maps
    beside the projectors, ``None`` where they are built in line.
    """

    def __init__(self) -> None:
        self.stages = _PreparationStages()
        self.symmetry_reduction = _Published()
        self.symmetry_maps: _OverlappedSymmetryMaps | None = None
        self._closers: list[Callable[[], None]] = []

    def join_on_exit(self, close: Callable[[], None]) -> None:
        """Have ``close()`` called when the preparation is left.

        ``close`` waits for a thread and returns nothing; it is called
        whether the thread has been joined before or not.
        """

        self._closers.append(close)

    def close(self) -> None:
        """Join every thread of the preparation, the last one started first.

        What a thread returned or raised is not read here.  A preparation
        that returns has joined its threads where their results are read,
        and one that raises reports its own error, which came first.  A
        thread that still waits for the symmetry reduction is told that
        there is none.
        """

        self.symmetry_reduction.publish(None)
        while self._closers:
            close = self._closers.pop()
            try:
                close()
            except Exception:
                pass


_IONIC_BACKEND_DEFAULT = "auto"


def ionic_backend_setting() -> str:
    """``PARSEC_IONIC_BACKEND``: ``native``, ``cupy`` or ``auto`` (unset: :data:`_IONIC_BACKEND_DEFAULT`, ``auto`` since
    2026-11-19; ``native`` before).

    ``native`` sums the local ionic potential and the atomic densities of a cluster with the C++/OpenMP builders on
    the host, in line, over every grid point.  ``cupy`` sums them on the GPU (``V_ion/cupy_ionic.py``), on a thread
    beside the sector set-up where ``PARSEC_OVERLAP_IONIC_SETUP`` allows, and there at one point per symmetry orbit
    where the images are exact (``PARSEC_IONIC_SUM_GRID``, ``PARSEC_IONIC_SUM_GROUP``).  ``auto`` takes ``cupy`` for
    a cluster where CuPy and a CUDA device are usable and the run is not a CPU run, ``native`` otherwise (a periodic
    cell builds its fields on its own routes either way).  Any other value is refused.  Measured on one A100 node
    (first runs, nanodiamond clusters, code defaults otherwise): the preparation of k06 took 3.8 s instead of
    16.0 s, k10 6.3 s instead of 196.5 s and k12 8.7 s instead of 434.0 s, the host sums of every grid point having
    taken 11.6, 189 and 420 s; the energies moved by 2e-9, 8e-8 and 1e-8 Ry (the round-off of the sums, through
    SCF loops to the criterion of those inputs).
    """

    value = os.environ.get("PARSEC_IONIC_BACKEND", "").strip().lower() or _IONIC_BACKEND_DEFAULT
    if value not in {"native", "cupy", "auto"}:
        raise ValueError("PARSEC_IONIC_BACKEND must be native, cupy or auto")
    return value


def _gpu_ionic_requested(*, periodic: bool = False, backend=None) -> bool:
    """Whether the ionic sums of a calculation go to the GPU builders (:func:`ionic_backend_setting`).

    ``backend`` is the execution backend asked for, where it is known: ``auto`` takes the GPU builders under
    ``auto`` and ``cupy`` alone (the builders are used where the CuPy backend is selected in any case).  The same
    answer on every MPI rank of a run: it reads the switch, the device policy and whether CuPy finds a device, not
    the rank.
    """

    value = ionic_backend_setting()
    if value != "auto":
        return value == "cupy"
    if periodic:
        return False
    if backend is not None and str(backend).strip().lower() not in {"auto", "cupy"}:
        return False
    from .resources import device_policy

    if device_policy() == "cpu" or not _cupy_installed():
        return False
    from .backends.cupy import cupy_available

    return bool(cupy_available())


def _ionic_overlap_requested(*, periodic: bool = False, backend=None) -> bool:
    """Whether the root-only ionic fields may leave the preparing thread.

    ``PARSEC_OVERLAP_IONIC_SETUP`` is on by default; ``0`` keeps the stages
    in line.  It applies to the GPU ionic fields
    (``PARSEC_IONIC_BACKEND=cupy``), whose host thread only waits for two
    kernels.  The C++/OpenMP builders work on the cores the host stages use
    and stay in line; an overlap of them was not measured.
    """

    return os.environ.get(
        "PARSEC_OVERLAP_IONIC_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"} and _gpu_ionic_requested(periodic=periodic, backend=backend)


def _ion_ion_overlap_requested() -> bool:
    """``PARSEC_OVERLAP_ION_ION_SETUP``: on unless 0, false, no or off.

    It acts where the ionic fields are built on their thread
    (``PARSEC_OVERLAP_IONIC_SETUP`` with the GPU ionic fields).  On, the
    ion-ion energy is summed on a thread of its own, started with that
    thread, and handed to it where its stage comes.  Off, the thread of the
    fields sums it after its densities, as before.  The energy is the value
    of the same function of the same atoms either way.
    """

    return os.environ.get(
        "PARSEC_OVERLAP_ION_ION_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


class _IonIonEnergyBeside:
    """The ion-ion energy of a system, summed on a thread of its own.

    The sum is a loop over the atoms in NumPy whose time grows with the
    square of their number.  It reads the atoms and the ionic charges and
    nothing else, touches no device and makes no MPI call, so the end of its
    thread leaves an open graph capture valid (``backends/cupy_capture.py``).
    The thread calls ``ion_ion_energy`` of the reference package with the
    atoms and pseudopotentials of the system: the energy is the one the
    stage of the reference preparation returns.

    It is started once the projectors of the reference preparation are
    built.  Beside that loop, which like this one runs in the interpreter,
    both were slower than one after the other on an A100 node; beside the
    symmetry maps the sum cost nothing.
    """

    def __init__(self, atoms, pseudopotentials, stages=None) -> None:
        self._atoms, self._pseudopotentials = atoms, pseudopotentials
        self.seconds = 0.0

        def evaluate():
            started = perf_counter()
            try:
                return ion_ion_energy(atoms, pseudopotentials)
            finally:
                self.seconds = perf_counter() - started
                if stages is not None:
                    stages.add("ion_ion_energy", started)

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-ion-ion",
        )
        self._future = self._executor.submit(evaluate)
        # The thread ends with its task, joined or not.
        self._executor.shutdown(wait=False)

    def energy(self, atoms, pseudopotentials) -> float:
        """Stand for ``ion_ion_energy`` in ``complete_single_point``.

        Waits for the thread and returns its value, or raises what it
        raised.  Asked for other atoms than those of the thread, it sums
        in place.
        """

        if atoms is not self._atoms or pseudopotentials is not self._pseudopotentials:
            return ion_ion_energy(atoms, pseudopotentials)
        return self._future.result()

    def close(self) -> None:
        """Wait for the thread without reading its value."""

        self._executor.shutdown(wait=True)


class _OverlappedIonicFields:
    """The root-only ionic fields, built on a thread beside host set-up.

    Only the XC evaluator and the SCF loop read the local ionic potential,
    the two densities and the ionic energies.  The thread runs
    ``complete_single_point`` of the reference package on a system that was
    prepared without them, so each field is the one the in-line stages
    return.  It makes no MPI call and takes no cuBLAS or cuSOLVER handle:
    its end leaves an open graph capture valid (``backends/cupy_capture.py``).

    The ion-ion energy, the last stage of that call, can be summed on a
    second thread meanwhile (:class:`_IonIonEnergyBeside`).  The thread of
    the fields takes its value where the stage comes, so both have ended
    when the fields are joined.
    """

    def __init__(self) -> None:
        self._executor = None
        self._future = None
        self._sum_devices = None
        self._stages = None
        self._ion_ion = None
        self.seconds = 0.0
        self.wait_seconds = 0.0
        # The devices the sums of the thread ran on, known at the join.
        self.devices: tuple[int, ...] = ()
        # The grid the sums of the thread ran on, in words, where the thread
        # was told how to choose it (PARSEC_IONIC_SUM_GRID), and the seconds
        # it took to choose before its first sum: its wait for the symmetry
        # reduction and the test of the images.  They are not in ``seconds``.
        self.sum_grid: str | None = None
        self.sum_grid_seconds = 0.0

    @property
    def started(self) -> bool:
        return self._future is not None

    def start(
        self,
        reference,
        *,
        sum_devices=None,
        stages=None,
        ion_ion_beside: bool = False,
        before_sums=None,
        after_sums=None,
        **builders,
    ) -> None:
        """Begin the fields of ``reference``, a system prepared without them.

        ``sum_devices`` is called at the join for the devices the sums of
        ``builders`` ran on.  ``stages`` records the thread and the wait at
        its join (:class:`_PreparationStages`).  With ``ion_ion_beside`` the
        ion-ion energy of ``reference`` is summed on a thread of its own.
        ``before_sums`` is called by the thread before its first sum and
        returns the grid the sums will run on, in words, for ``details``.
        ``after_sums`` is called by the thread once its sums have returned
        or raised: it takes back what ``before_sums`` told the builders.
        """

        if self._future is not None:
            raise RuntimeError("the ionic fields of a preparation start once")
        self._sum_devices = sum_devices
        self._stages = stages
        if ion_ion_beside:
            self._ion_ion = _IonIonEnergyBeside(
                reference.atoms, reference.pseudopotentials, stages
            )
            builders = {**builders, "ion_ion_energy_builder": self._ion_ion.energy}

        def build():
            started = perf_counter()
            first_sum = started
            try:
                try:
                    if before_sums is not None:
                        self.sum_grid = before_sums()
                        first_sum = perf_counter()
                        self.sum_grid_seconds = first_sum - started
                    complete = complete_reference_single_point(reference, **builders)
                finally:
                    if after_sums is not None:
                        after_sums()
            finally:
                if stages is not None:
                    stages.add("ionic_fields", started)
            # The seconds of the fields, counted from their first sum as
            # they were before the thread chose a grid: a thread that waits
            # for the symmetry reduction does no work that the wait hides.
            return complete, perf_counter() - first_sum

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-ionic-setup",
        )
        self._future = self._executor.submit(build)
        # Where a failure of the caller leaves the thread unjoined, it still
        # ends with its task and not when the executor is collected.
        self._executor.shutdown(wait=False)

    def result(self):
        """Join the thread and return the complete reference system.

        Its ``timings.total_seconds`` is still the wall time of the
        preparation that left the fields out.  The thread time and the wait
        here are in no stage of the calling thread; ``details`` reports them.
        """

        started = perf_counter()
        try:
            complete, self.seconds = self._future.result()
        finally:
            self._executor.shutdown(wait=True)
            if self._ion_ion is not None:
                # Its value was taken by the thread of the fields, unless
                # that thread failed before the stage.
                self._ion_ion.close()
            self.wait_seconds = perf_counter() - started
            if self._stages is not None:
                self._stages.add("ionic_fields_wait", started)
        if self._ion_ion is not None:
            # The stage of the thread of the fields only took the value: the
            # seconds of the sum are those of its own thread.
            complete = replace(
                complete,
                timings=replace(
                    complete.timings, ion_ion_seconds=self._ion_ion.seconds
                ),
            )
        if self._sum_devices is not None:
            self.devices = tuple(self._sum_devices())
        return complete

    def close(self) -> None:
        """Wait for the threads without reading what they built or raised.

        For a preparation that is left before the join, or after it.
        """

        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._ion_ion is not None:
            self._ion_ion.close()

    def details(self) -> tuple[tuple[str, str], ...]:
        """Report the time of the fields, the wait at the join and what was hidden.

        The time of the fields runs from the first sum of the thread to its
        end.  What the thread took before that to choose the grid of its
        sums has an entry of its own.
        """

        if self._future is None:
            return (("ionic_setup", "inline"),)
        return (
            ("ionic_setup", "overlapped with symmetry and orbital setup"),
            ("ionic_setup_seconds", f"{self.seconds:.6f}"),
            ("ionic_setup_wait_seconds", f"{self.wait_seconds:.6f}"),
            (
                "ionic_setup_overlapped_seconds",
                f"{max(0.0, self.seconds - self.wait_seconds):.6f}",
            ),
            # PARSEC_IONIC_SUM_GRID: the points the sums of the thread ran
            # on, and the seconds the thread took to choose them (its wait
            # for the symmetry reduction and the test of the images).
            *(
                ()
                if self.sum_grid is None
                else (
                    ("ionic_sum_grid", self.sum_grid),
                    (
                        "ionic_sum_grid_choice_seconds",
                        f"{self.sum_grid_seconds:.6f}",
                    ),
                )
            ),
            # PARSEC_OVERLAP_ION_ION_SETUP: where the ion-ion energy was
            # summed, and for a thread of its own the seconds it took.
            *(
                (("ion_ion_setup", "last stage of the ionic thread"),)
                if self._ion_ion is None
                else (
                    ("ion_ion_setup", "own thread beside the ionic fields"),
                    ("ion_ion_setup_seconds", f"{self._ion_ion.seconds:.6f}"),
                )
            ),
        )


def ionic_sum_group_setting() -> str:
    """``PARSEC_IONIC_SUM_GROUP``: ``full`` (the default since 2026-11-19) or ``first``.

    The GPU ionic sums on their thread run at one point per orbit of the reduction the preparation publishes
    (``PARSEC_IONIC_SUM_GRID=auto``, :func:`_choose_ionic_sum_grid`).  ``full``: where the weighted sectors find
    the point group of the structure (a :class:`.Symmetry.PointGroupReduction`), the orbits of that full group
    (:mod:`.Symmetry.field_orbits`; Td: N/24 points instead of the N/4 of the abelian D2), published once the group
    is found; any group that maps the atoms and the grid points onto themselves in every bit may stand for its
    orbits, whichever group the orbitals take.  Where the atoms are images under that group only within the
    detection tolerance, the sums try the orbits of the sectors' group ``H`` and then those of the abelian group
    detected first (:func:`_ionic_sum_candidates`) before every grid point.  ``first``: the reduction detected first
    (the abelian group of the thread of the maps), published at once, as before.
    """

    value = os.environ.get("PARSEC_IONIC_SUM_GROUP", "full").strip().lower() or "full"
    if value not in {"full", "first"}:
        raise ValueError("PARSEC_IONIC_SUM_GROUP must be full or first")
    return value


def _ionic_sum_reduction(reduction, stages=None):
    """The reduction whose orbits the ionic sums take for ``reduction``: the orbits of the full group of a point-group
    reduction (:func:`.Symmetry.field_orbits.field_orbit_reduction`, built once and kept for the SCF fields), else
    ``reduction`` itself."""

    if reduction is None or not hasattr(reduction, "coset_images"):
        return reduction
    from .Symmetry.field_orbits import field_orbit_reduction

    started = perf_counter()
    try:
        field = field_orbit_reduction(reduction)
    except (ValueError, RuntimeError):
        return reduction
    finally:
        if stages is not None:
            stages.add("field_orbits", started)
    return reduction if field is None else field


def _ionic_sum_candidates(reduction, stages=None, first=None):
    """What the preparation publishes for the ionic sums of ``reduction`` (``PARSEC_IONIC_SUM_GROUP=full``).

    The reductions whose orbits the sums try, largest group first: the orbits of the full group of a point-group
    reduction (:func:`_ionic_sum_reduction`), the reduction itself (its subgroup ``H``) and ``first``, the abelian
    group detected first, where it is another.  A structure may map onto itself exactly under the sign flips of the
    abelian group but only within the detection tolerance under the axis permutations of the full group
    (coordinates with last-bit differences between the axes): its sums then keep the orbits they took before
    instead of every grid point (:func:`_choose_ionic_sum_grid`).  One reduction alone is published as itself, two or
    more as a tuple; ``None`` without any.
    """

    chain: list = []
    for item in (_ionic_sum_reduction(reduction, stages), reduction, first):
        if item is not None and all(item is not other for other in chain):
            chain.append(item)
    if not chain:
        return None
    return chain[0] if len(chain) == 1 else tuple(chain)


def _orbit_sum_choice(grid, atoms, candidates, *, atom_tolerance: float = 0.0):
    """The first of ``candidates`` (a reduction, a tuple of them, or ``None``) whose images of ``atoms`` and of the
    grid points are exact (:func:`.Symmetry.exact_images.orbit_sum_points`, ``atom_tolerance`` for moved atoms).

    Returns ``(reduction, points, obstacles)``: the reduction and its representative points, or ``(None, None,
    obstacles)`` where none passes, with why each failed, in their order.
    """

    from .Symmetry.exact_images import orbit_sum_points

    if candidates is None:
        return None, None, []
    if not isinstance(candidates, tuple):
        candidates = (candidates,)
    # (the tolerance named only for moved atoms: the test of a first structure is called as it always was)
    tolerance = {"atom_tolerance": atom_tolerance} if atom_tolerance > 0.0 else {}
    obstacles = []
    for reduction in candidates:
        if reduction is None:
            continue
        points, obstacle = orbit_sum_points(grid, atoms, reduction, **tolerance)
        if points is not None:
            return reduction, points, obstacles
        obstacles.append(obstacle)
    return None, None, obstacles


def _orbit_sum_wedge_text(reduction, obstacles) -> str:
    """The backend report of sums on the orbits of ``reduction`` after the larger groups whose ``obstacles`` failed."""

    text = f"symmetry wedge ({int(reduction.wedge_size)} of {int(reduction.full_size)} points)"
    if obstacles:
        text += f"; a group of {int(reduction.group_order)} operations, the larger one failed: {obstacles[0]}"
    return text


def _choose_ionic_sum_grid(builders, reference, preparation) -> str:
    """Let the ionic sums of ``builders`` run on the symmetry wedge where that is exact.

    Called by the thread of the ionic fields before its first sum.  The
    local ionic potential and the atomic densities are sums of one term per
    atom, totally symmetric under the group of the structure, and the SCF
    loop of a symmetric structure keeps one value per orbit of them.  Where
    ``PARSEC_IONIC_SUM_GRID`` allows it (``auto``, the default) and
    ``Symmetry/exact_images.py`` finds the images of the atoms and of the
    grid points under the detected group exact in every bit, the builders
    are told to sum at the representative point of every orbit and to give
    its value to the whole orbit.  In every other case they sum every grid
    point, as before, and the text returned says why.  The wedge is an
    economy and the full grid is always right: a test that cannot be made,
    or builders that refuse the orbits, are one more such case and do not
    fail the preparation.

    The thread waits here until the reduction of the preparation is known:
    the thread of the symmetry maps publishes it when it is detected, the
    thread that prepares where it detects in line, and a preparation that
    looks for no symmetry, or fails, publishes that there is none.  With
    ``PARSEC_IONIC_SUM_GROUP=full`` the published value may be a tuple of
    reductions (:func:`_ionic_sum_candidates`): the sums take the first
    whose images are exact.
    """

    if ionic_sum_grid_setting() == "full":
        return "full grid (PARSEC_IONIC_SUM_GRID=full)"
    restrict = getattr(builders, "sum_on_orbits", None)
    if restrict is None:
        return "full grid (these builders sum every point)"
    stages = preparation.stages
    started = perf_counter()
    published = preparation.symmetry_reduction.wait()
    stages.add("ionic_symmetry_wait", started)
    if published is None or (isinstance(published, tuple) and all(item is None for item in published)):
        return "full grid (no symmetry reduction)"

    started = perf_counter()
    try:
        reduction, points, obstacles = _orbit_sum_choice(reference.grid, reference.atoms, published)
        if points is None:
            return f"full grid ({obstacles[0] if obstacles else 'no symmetry reduction'})"
        wedge = _orbit_sum_wedge_text(reduction, obstacles)
        restrict(reference.grid, reduction, points)
        return wedge
    except Exception as error:
        # Whatever the builders were told before they refused is taken back.
        every_point = getattr(builders, "sum_on_full_grid", None)
        if every_point is not None:
            every_point()
        # An entry of the backend report: the first words of the error.
        words = f"{type(error).__name__}: {error}"[:100]
        return f"full grid (the test of the images failed: {words})"
    finally:
        stages.add("ionic_sum_grid_test", started)


def _ionic_field_overlap(mpi_context, *, periodic: bool = False, backend=None) -> _OverlappedIonicFields | None:
    """Return the overlap of the ionic fields for a rank that may use one.

    The root rank of an MPI run and a process without one build the fields.
    Where they are GPU kernels and the overlap is requested, they run beside
    the host set-up that follows the reference preparation and are joined at
    their first reader.  ``None`` keeps them in line.
    """

    builds_fields = mpi_context is None or mpi_context.rank == mpi_context.root
    if builds_fields and _ionic_overlap_requested(periodic=periodic, backend=backend):
        return _OverlappedIonicFields()
    return None


def _reference_cluster_grid(settings):
    """Call the grid builder the reference preparation takes when given none."""

    from parsec_python.SCF import single_point

    return single_point.build_cluster_grid(settings)


def _timed_builders(options: dict, stages: _PreparationStages) -> dict:
    """Return ``options`` with the builders of the reference system on the clock.

    Each builder hook of ``prepare_single_point`` and
    ``complete_single_point`` of the reference package that ``options``
    names is called through ``stages``, which records its start and end
    under the name of what it builds.  The grid builder of the reference
    package is named where the options leave the grid to it, so that the
    grid has a stage whichever builder makes it.
    """

    timed = dict(options)
    timed["grid_builder"] = stages.timed(
        "grid", options.get("grid_builder", _reference_cluster_grid)
    )
    for hook, name in (
        ("negative_laplacian_builder", "finite_difference"),
        ("local_ionic_builder", "local_ionic_potential"),
        ("nonlocal_projector_builder", "projectors"),
    ):
        if hook in options:
            timed[hook] = stages.timed(name, options[hook])
    if "atomic_density_builder" in options:
        timed["atomic_density_builder"] = stages.timed(
            lambda *_arguments, core=False, **_options: (
                "core_density" if core else "initial_density"
            ),
            options["atomic_density_builder"],
        )
    return timed


def _grid_builder_starting_the_maps(build, problem, selection, symmetry_maps, point_group=False):
    """Return ``build`` followed by the start of the symmetry maps of its grid.

    The reference preparation calls its grid builder once the atoms are
    placed and the electrons counted, and builds the finite differences and
    the projectors after it.  The maps are started on the grid it returns
    and on the atoms as that preparation places them, by its own
    expression, and with the representation phases where the selected
    backend solves in symmetry sectors.
    """

    from parsec_python.V_ion import center_cluster_geometry

    def build_grid(settings):
        grid = build(settings)
        if not symmetry_maps.started:
            atoms = (
                center_cluster_geometry(problem.atoms)
                if problem.recenter_geometry
                else tuple(problem.atoms)
            )
            symmetry_maps.start(
                grid, atoms, decompose=selection.selected == "cupy",
                point_group=point_group and selection.selected == "cupy",
            )
        return grid

    return build_grid


def _prepare_reference_physics(
    problem: SinglePointInput,
    selection: BackendSelection,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
    orbital_operators_only: bool = False,
    ionic_sector_rank: bool = False,
    ionic_report: dict | None = None,
    ionic_field_overlap: _OverlappedIonicFields | None = None,
    preparation: _Preparation | None = None,
):
    """Build the static physics with the builders of the selected backend.

    ``ionic_sector_rank`` tells the GPU ionic sums that this process is an
    MPI rank whose sector groups cover its devices (see
    :func:`V_ion.cupy_ionic.ionic_device_ids`).  ``ionic_report`` receives
    under ``devices`` the devices those sums ran on; it is left as it is
    where none ran here: another ionic backend, a rank that builds no ionic
    field, a reference taken from the resident cache, or fields left to
    ``ionic_field_overlap``.  The thread of the overlap sums after this call
    has returned, on the devices the process setting gives the calling
    thread, and the overlap holds the devices once it is joined.

    ``preparation`` is the preparation this call is a stage of.  With it the
    builders of the reference system are called through its clock, which
    records when each began and ended; they build what they build without.
    Where it has a thread for the symmetry maps, that thread is started as
    soon as the grid exists, provided the reference system is built here
    by the native builders, without ionic fields (a rank that builds none,
    or fields left to ``ionic_field_overlap``) and without the full-grid
    matrix (``defer_native_laplacian``): its host work after the grid is
    then the projectors, beside which the maps were measured.
    """

    cache_capacity = _resident_reference_cache_size()
    # A forced fresh model prediction must not be bypassed by the warmed
    # prepared-system cache.  The ordinary exact-key ML prediction cache still
    # makes repeat calculations inexpensive when ``regenerate`` is false.
    initial_settings = getattr(problem, "initial_density_settings", None)
    if initial_settings is not None and initial_settings.regenerate:
        cache_capacity = 0
    # A system without ionic fields and densities must not be stored under
    # the key of a complete one.
    if orbital_operators_only:
        cache_capacity = 0
    reduced_options = (
        {"orbital_operators_only": True} if orbital_operators_only else {}
    )
    # PARSEC_FAST_GRID: the same grid arrays without the cube of triples.
    if fast_grid_requested():
        reduced_options["grid_builder"] = build_cluster_grid_by_slabs
    # The static atomic tail of the Hartree boundary is built for the
    # Hartree solver of the selected backend, beside its boundary geometry.
    reduced_options["build_atomic_tail"] = False
    cache_started = perf_counter()
    cache_key = None
    if cache_capacity:
        cache_key = _reference_cache_key(
            problem,
            selection,
            defer_native_laplacian=defer_native_laplacian,
            cache_directory=deferred_laplacian_cache_directory,
        )
        with _REFERENCE_CACHE_LOCK:
            cached_reference = _REFERENCE_CACHE.get(cache_key)
            if cached_reference is not None:
                _REFERENCE_CACHE.move_to_end(cache_key)
        if cached_reference is not None:
            return _cached_reference_view(
                cached_reference,
                cache_directory=deferred_laplacian_cache_directory,
                lookup_seconds=perf_counter() - cache_started,
            )

    if getattr(problem, "periodic_cell", None) is not None:
        if orbital_operators_only and _periodic_operators_only_available(problem):
            # A rank that only shares the basis of the sector: grid, kinetic
            # operator and projectors alone.
            return _prepare_periodic_orbital_operators(problem)
        from .Laplacian import periodic_laplacian_builder

        # the full-grid matrix of a box built when something reads it (PARSEC_PERIODIC_DEFER_LAPLACIAN): the sector
        # stencils come from the grid
        reference = prepare_reference_periodic_single_point(
            problem, negative_laplacian_builder=periodic_laplacian_builder()
        )
        if cache_key is not None:
            _remember_reference(cache_key, reference, cache_capacity)
        return reference

    # Static construction and repeated Hamiltonian execution are independent
    # choices.  In the default hybrid path C++ builds the compressed-grid
    # finite-difference CSR once, then CuPy owns the repeated H@Q operations.
    if selection.finite_difference_builder == "native":
        from .backends.native import _load_native, build_native_negative_laplacian
        from .V_ion import NativeIonicBuilders

        try:
            native_module = _load_native()
        except BackendUnavailableError:
            # Selection normally establishes availability first.  Keeping
            # this optional optimization guard also makes the finite-
            # difference builder independently testable and supports older
            # extension builds that predate the radial kernel.
            native_module = None
        ionic_builders = (
            NativeIonicBuilders()
            if native_module is not None
            and hasattr(native_module, "RadialGridEvaluator")
            else None
        )
        gpu_ionic_fields = False
        if (ionic_builders is not None and selection.selected == "cupy"
                and _gpu_ionic_requested()):
            from .V_ion.cupy_ionic import CupyIonicBuilders

            ionic_builders = CupyIonicBuilders(sector_rank=ionic_sector_rank)
            gpu_ionic_fields = True

        if defer_native_laplacian:
            from .Laplacian import DeferredNativeNegativeLaplacian

            def laplacian_builder(grid):
                return DeferredNativeNegativeLaplacian(
                    grid,
                    cache_directory=deferred_laplacian_cache_directory,
                )
        else:
            laplacian_builder = build_native_negative_laplacian
        builder_options = {
            "negative_laplacian_builder": laplacian_builder,
            **reduced_options,
        }
        if ionic_builders is not None:
            builder_options.update(
                local_ionic_builder=(
                    ionic_builders.build_local_ionic_potential
                ),
                nonlocal_projector_builder=(
                    ionic_builders.build_nonlocal_projectors
                ),
                atomic_density_builder=(
                    ionic_builders.superpose_atomic_density
                ),
            )
        fields_on_their_thread = (
            ionic_field_overlap is not None
            and gpu_ionic_fields
            and not orbital_operators_only
            # A resident process stores the complete system under its key.
            and not cache_capacity
        )
        if preparation is not None:
            builder_options = _timed_builders(builder_options, preparation.stages)
            if (
                preparation.symmetry_maps is not None
                and (fields_on_their_thread or orbital_operators_only)
                and defer_native_laplacian
            ):
                # This preparation builds no ionic field and no full-grid
                # matrix: after the grid its host work is the projectors,
                # which the maps may run beside.
                builder_options["grid_builder"] = _grid_builder_starting_the_maps(
                    builder_options["grid_builder"],
                    problem,
                    selection,
                    preparation.symmetry_maps,
                )
        if fields_on_their_thread:
            reference = prepare_reference_single_point(
                problem, **builder_options, orbital_operators_only=True
            )
            # A thread starts with device 0 current, whatever the caller's.
            ionic_builders.keep_current_device()
            ionic_field_overlap.start(
                reference,
                sum_devices=lambda: ionic_builders.device_ids,
                **({} if preparation is None else {"stages": preparation.stages}),
                # The projectors are built: the loop of the ion-ion energy
                # no longer competes with theirs for the interpreter.
                ion_ion_beside=_ion_ion_overlap_requested(),
                # The thread chooses the points of its sums once the
                # symmetry of the preparation is known.  The orbits were
                # tested for the atoms of this system and for no others:
                # the builders sum every point again once the fields of the
                # thread are built.
                **(
                    {}
                    if preparation is None
                    else {
                        "before_sums": lambda: _choose_ionic_sum_grid(
                            ionic_builders, reference, preparation
                        ),
                        "after_sums": getattr(
                            ionic_builders, "sum_on_full_grid", None
                        ),
                    }
                ),
                local_ionic_builder=builder_options["local_ionic_builder"],
                atomic_density_builder=builder_options["atomic_density_builder"],
                build_atomic_tail=False,
            )
            return reference
        reference = prepare_reference_single_point(problem, **builder_options)
        ionic_devices = getattr(ionic_builders, "device_ids", ())
        if ionic_report is not None and ionic_devices:
            ionic_report["devices"] = tuple(ionic_devices)
        if cache_key is not None:
            _remember_reference(cache_key, reference, cache_capacity)
            negative_laplacian = reference.negative_laplacian
            if defer_native_laplacian:
                negative_laplacian.reference_static_cache_status = "miss-stored"
                negative_laplacian.reference_static_cache_lookup_seconds = (
                    perf_counter() - cache_started
                )
        return reference
    if selection.finite_difference_builder != "reference":
        raise RuntimeError(
            "unhandled finite-difference builder "
            f"{selection.finite_difference_builder!r}"
        )
    if preparation is not None:
        reduced_options = _timed_builders(reduced_options, preparation.stages)
    reference = prepare_reference_single_point(problem, **reduced_options)
    if cache_key is not None:
        _remember_reference(cache_key, reference, cache_capacity)
    return reference


def _periodic_operators_only_available(problem) -> bool:
    """Whether :func:`_prepare_periodic_orbital_operators` builds what the full preparation would of ``problem``.

    Andrew Grieder's routines of a bulk cell on the orthorhombic grid; the
    Fortran-derived routines, wires and slabs keep the full preparation.
    """

    grid = problem.grid
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    return (
        not getattr(grid, "fortran_routines", False)
        and int(getattr(grid, "periodicity", 3)) == 3
        and int(getattr(problem.periodic_cell, "dimensions", 3)) == 3
        and np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10)
    )


def _prepare_periodic_orbital_operators(problem):
    """The orbital operators of a bulk cell for a rank that shares the basis of its sector.

    The steps of :func:`parsec_python.SCF.pbc.prepare_periodic_single_point`
    that make ``H = -nabla_FD^2 + diag(V) + V_NL`` apart from ``V``, through
    the same functions of that module and with the same arguments: the
    pseudopotentials and the electron count, the grid, the finite-difference
    Laplacian and the projectors with their lattice images.  The kinetic
    operator and the projectors are those of the full preparation element for
    element.  What only the SCF loop of the root reads (the gradient matrices,
    the local ionic potential, the starting and core densities, the Ewald
    energies) is not built: ``None``, and NaN for the energies.
    """

    from parsec_python.SCF import pbc as periodic

    started = perf_counter()
    atoms = tuple(problem.atoms)
    stage = perf_counter()
    pseudopotentials = periodic.load_pseudopotentials(
        problem.pseudopotentials, xc_functional=problem.scf.xc_functional
    )
    pseudopotential_seconds = perf_counter() - stage
    electron_count = periodic.ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    if electron_count <= 0:
        raise ValueError("the requested system has no valence electrons")
    stage = perf_counter()
    grid = periodic.build_periodic_grid(problem.periodic_cell, problem.grid)
    grid_seconds = perf_counter() - stage
    # the tables of the full preparation (PARSEC_LOCAL_FILTER), for whatever a rank reads of the local potential
    pseudopotentials = periodic.apply_local_filters(pseudopotentials, problem.pseudopotentials, grid)
    lattice_vectors = problem.periodic_cell.lattice_vectors
    # An atom outside the cell (a relax_restart.dat or mdinit.dat of an atom that crossed a face) is folded into it
    # for the projectors, as the full preparation folds it for its builders (pbc.atoms_for_cell_builders); the
    # system keeps the coordinates as given.
    builder_atoms = periodic.atoms_for_cell_builders(lattice_vectors, atoms)
    stage = perf_counter()
    from .Laplacian import periodic_laplacian_builder

    negative_laplacian = (periodic_laplacian_builder() or periodic.build_negative_laplacian)(grid)
    finite_difference_seconds = perf_counter() - stage
    stage = perf_counter()
    nonlocal_operator = periodic.build_nonlocal_projectors(
        grid, builder_atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, periodic_dimensions=3
    )
    nonlocal_seconds = perf_counter() - stage
    return periodic.PeriodicPreparedSinglePointSystem(
        input=problem,
        atoms=atoms,
        electron_count=electron_count,
        pseudopotentials=pseudopotentials,
        grid=grid,
        negative_laplacian=negative_laplacian,
        gradient=None,
        ionic_potential=None,
        nonlocal_operator=nonlocal_operator,
        initial_density=None,
        core_density=None,
        ion_ion_energy=float("nan"),
        atomic_reference_correction=float("nan"),
        alpha_z_energy=float("nan"),
        timings=PreparationTimings(
            pseudopotential_loading_seconds=pseudopotential_seconds,
            grid_seconds=grid_seconds,
            finite_difference_seconds=finite_difference_seconds,
            nonlocal_ionic_seconds=nonlocal_seconds,
            total_seconds=perf_counter() - started,
        ),
    )


def _build_backend(
    reference,
    selection: BackendSelection,
    *,
    defer_cupy_device_operator: bool = False,
    nonlocal_operator=None,
):
    """The Hamiltonian backend of ``selection``.

    ``nonlocal_operator`` is the separable factor of the device
    Hamiltonians where it is not the KB factor of ``reference`` (the LDA+U
    columns appended, Hamiltonian.hubbard_device).
    """

    common = {
        "requested": selection.requested,
        "fallback_reasons": selection.fallback_reasons,
    }
    if nonlocal_operator is None:
        nonlocal_operator = reference.nonlocal_operator
    if selection.selected == "scipy":
        return ScipyHamiltonianBackend(
            reference.negative_laplacian,
            nonlocal_operator,
            **common,
        )
    if selection.selected == "native":
        from .backends.native import NativeHamiltonianBackend

        return NativeHamiltonianBackend(
            reference.negative_laplacian,
            nonlocal_operator,
            **common,
        )
    if selection.selected == "cupy":
        from .backends.cupy_runtime import CuPyHamiltonianBackend

        return CuPyHamiltonianBackend(
            reference.negative_laplacian,
            nonlocal_operator,
            defer_device_operator=defer_cupy_device_operator,
            **common,
        )
    raise RuntimeError(f"unhandled backend selection {selection.selected!r}")


def _normalize_symmetry_mode(value: SymmetryMode | str) -> SymmetryMode:
    """Validate the public auto/on/off symmetry policy."""

    normalized = str(value).strip().lower()
    if normalized not in {"auto", "on", "off"}:
        raise ValueError("symmetry must be one of 'auto', 'on', or 'off'")
    return normalized  # type: ignore[return-value]


def _root_rank_only(component: str):
    """Return a hook that refuses to run ``component`` on a sector worker.

    A rank that only serves symmetry sectors prepares no Hartree or XC
    objects.  Failing with the reason is preferable to an attribute error on
    a missing field if the SCF loop is ever entered there.
    """

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            f"{component} exists on the MPI root rank only; this rank "
            "prepared symmetry-sector operators"
        )

    return unavailable


def _hartree_device_request() -> int | str | None:
    """Parse ``PARSEC_HARTREE_DEVICE``.

    ``auto``, the default, or a CuPy device index of this process puts the
    GPU Hartree objects together on one device.  ``off`` returns ``None`` and
    leaves them on their default devices: the CG on the first sector device,
    the boundary geometry on the device current in the thread that builds it.
    """

    raw = os.environ.get("PARSEC_HARTREE_DEVICE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw == "off":
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_HARTREE_DEVICE must be auto, off or a CuPy device index"
        ) from error
    if value < 0:
        raise ValueError("PARSEC_HARTREE_DEVICE cannot be negative")
    return value


def _resident_hartree_request() -> str:
    """Parse ``PARSEC_CUPY_RESIDENT_HARTREE``: ``auto`` (the default), ``1`` or ``0``.

    ``auto`` (the default since 2026-11-07) runs the device-resident Hartree
    chain (``Hartree/cupy_resident.py``) where the conjugate gradients run on
    a GPU on the symmetry wedge of the SCF fields, with the boundary values
    built on that device; ``1`` asks for the chain by name and stops where it
    does not apply or cannot be built; ``0`` keeps the host interface, which
    was the default before.
    """

    raw = os.environ.get("PARSEC_CUPY_RESIDENT_HARTREE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw not in {"0", "1"}:
        raise ValueError("PARSEC_CUPY_RESIDENT_HARTREE must be auto, 1 or 0")
    return raw


def _resident_hartree_plan(
    request: str,
    *,
    linear_backend: str,
    hartree_reduction,
    orbital_decomposition,
    orbital_reduction,
    boundary_builder_planned: bool,
) -> tuple[bool, str]:
    """Whether the preparation builds the resident Hartree chain, and why.

    Decided before the boundary thread starts, because the chain takes its
    boundary from a GPU builder on the wedge where the host builder would
    otherwise use its orbit table.  The chain applies where the conjugate
    gradients run on a GPU (``linear_backend``), on the wedge
    ``hartree_reduction`` that the SCF fields use too (a symmetry-sector
    eigensolver, whose reduction is ``orbital_reduction``), with a multipole
    boundary builder (``boundary_builder_planned``), on a grid of ``int32``
    indices, and where neither ``PARSEC_HARTREE_BOUNDARY_BACKEND=native`` nor
    ``PARSEC_HARTREE_BOUNDARY_KERNEL=full`` keeps the boundary on the host by
    name.  Returns ``(applies, reason)``; ``reason`` completes the
    ``hartree_resident_chain`` line.
    """

    if request == "0":
        return False, "PARSEC_CUPY_RESIDENT_HARTREE=0"
    prefix = "auto" if request == "auto" else "PARSEC_CUPY_RESIDENT_HARTREE=1"
    if linear_backend != "cupy":
        return False, f"{prefix}: the conjugate gradients run on the host"
    if hartree_reduction is None:
        return False, f"{prefix}: the Hartree problem is solved on the full grid"
    if orbital_decomposition is None:
        return False, (
            f"{prefix}: the orbitals are solved on the full grid, whose SCF "
            "fields have no symmetry wedge"
        )
    if orbital_reduction is not hartree_reduction:
        return False, f"{prefix}: the SCF fields use another wedge than the Hartree solve"
    if not boundary_builder_planned:
        return False, f"{prefix}: no multipole boundary builder (domain or boundary method)"
    if int(hartree_reduction.full_size) > np.iinfo(np.int32).max:
        return False, f"{prefix}: the grid exceeds int32 indices"
    if os.environ.get("PARSEC_HARTREE_BOUNDARY_BACKEND", "").strip().lower() == "native":
        return False, f"{prefix}: PARSEC_HARTREE_BOUNDARY_BACKEND=native keeps the boundary on the host"
    if _hartree_boundary_kernel() == "full" and os.environ.get(
        "PARSEC_HARTREE_BOUNDARY_BACKEND", ""
    ).strip().lower() != "cupy":
        return False, f"{prefix}: PARSEC_HARTREE_BOUNDARY_KERNEL=full keeps the boundary on the host"
    return True, f"{prefix}: the GPU conjugate gradients on the symmetry wedge of the SCF fields"


class _GPUHartreeUnavailable(Exception):
    """The GPU CG chosen by ``auto`` could not be built on the symmetry wedge."""


def _hartree_linear_backend(
    selection: BackendSelection, *, periodic: bool
) -> tuple[str, str]:
    """Choose the conjugate gradients of the cluster Hartree solve.

    Returns ``(backend, reason)``: ``backend`` is ``"cupy"`` for the FP64
    CG on a GPU (``Hartree/cupy_prepared.py``) and ``"native"`` for the
    CG of the selected Hartree backend (the C++/OpenMP CG of the native
    components; the SciPy CG or the CG of ``--backend cupy`` where those are
    selected, which this choice does not touch).  Both run the same
    recurrence with the same stopping test ``||r_k|| <= rtol ||r_0|| +
    atol``, matrix-vector budget and chronological starting vector; they
    differ in the order of the floating-point sums, so the potential differs
    within the CG tolerance.

    ``PARSEC_HARTREE_LINEAR_BACKEND``: unset or ``auto`` (the default since
    2026-11-07) takes the GPU CG where it applies, the CuPy orbital backend
    with the native Hartree components (``--backend auto`` on a GPU) for an
    isolated system, and the CG of the selection elsewhere; ``native`` keeps
    the host CG; ``cupy`` asks for the GPU CG by name and is refused where it
    does not apply.  A periodic cell is solved by its own periodic solver
    whatever the variable says.
    """

    requested = os.environ.get("PARSEC_HARTREE_LINEAR_BACKEND", "").strip().lower()
    if requested not in {"", "auto", "native", "cupy"}:
        raise ValueError(
            "PARSEC_HARTREE_LINEAR_BACKEND must be auto, native or cupy"
        )
    if periodic:
        # The variable chooses the CG of the isolated-cluster Hartree solver.
        return "native", "periodic cell: the periodic Hartree solver"
    applies = selection.selected == "cupy" and selection.hartree_backend == "native"
    if requested == "cupy":
        if not applies:
            raise ValueError(
                "PARSEC_HARTREE_LINEAR_BACKEND=cupy requires the auto/hybrid CUDA "
                "backend with native boundary construction"
            )
        return "cupy", "PARSEC_HARTREE_LINEAR_BACKEND=cupy"
    if requested == "native":
        return "native", "PARSEC_HARTREE_LINEAR_BACKEND=native"
    if applies:
        return "cupy", "auto: the GPU of the orbitals"
    return "native", (
        f"auto: no GPU CG beside the {selection.selected} orbitals and the "
        f"{selection.hartree_backend} Hartree components"
    )


def _hartree_linear_report(
    selection: BackendSelection, *, periodic: bool, backend: str, reason: str
) -> str:
    """The ``hartree_linear_backend`` line: the CG that runs and why.

    Outside the native Hartree components of CuPy or CPU orbitals the
    variable chooses nothing, and the line names the solver of the
    selection instead of ``native``.
    """

    if periodic:
        return "periodic (the solver of the periodic cell; PARSEC_HARTREE_LINEAR_BACKEND acts on clusters)"
    if selection.hartree_backend == "scipy":
        return "scipy (the SciPy CG of the selected backend; PARSEC_HARTREE_LINEAR_BACKEND acts on the native components)"
    if selection.hartree_backend == "cupy":
        return "cupy-csr (the device CSR CG of --backend cupy; PARSEC_HARTREE_LINEAR_BACKEND acts on the native components)"
    if selection.selected != "cupy":
        return f"native (the C++/OpenMP CG of a CPU run; {reason})"
    return f"{backend} ({reason})"


def _gpu_hartree_device_bytes(poisson_solver, boundary_builder, resident_hartree=None) -> dict[int, int]:
    """Device bytes of the GPU Hartree objects, by device (empty for host objects)."""

    counts: dict[int, int] = {}
    backend = getattr(getattr(poisson_solver, "solver", poisson_solver), "backend", None)
    if type(backend).__name__ == "CuPyPreparedConjugateGradientBackend":
        counts[int(backend.device_id)] = int(backend.device_bytes)
    if type(boundary_builder).__name__ in _GPU_BOUNDARY_BUILDERS:
        device = getattr(boundary_builder, "device_id", None)
        stored = getattr(boundary_builder, "device_storage_bytes", None)
        if device is not None and stored:
            counts[int(device)] = counts.get(int(device), 0) + int(stored)
    stored = getattr(resident_hartree, "device_bytes", None)
    device = getattr(resident_hartree, "device_id", None)
    if stored and device is not None:
        counts[int(device)] = counts.get(int(device), 0) + int(stored)
    return counts


class _ResidentHartreeSolve:
    """The Hartree solver of a prepared system with the resident chain in front.

    Every solve runs ``chain`` (:class:`.Hartree.cupy_resident.CuPyResidentHartree`)
    and adds its seconds to ``statistics``; the result carries the atomic
    tail in rows that ``tail()`` returns, as the host interface's does.
    Where a device allocation of a solve fails (CuPy's ``OutOfMemoryError``,
    or a CUDA runtime or driver error of status 2; :func:`allocation_failure`)
    and the chain was not asked for by name, the chain ends for the rest of
    the calculation and the route before it takes over: the arrays of the
    chain are dropped, ``leave()`` replaces its GPU boundary builder by the
    host builder of that wedge (the orbit table) and returns the report lines
    of what was built, the predictor history goes to the host solver
    ``poisson_solver`` (the GPU CG stays), and ``host_solve``, the host
    interface, repeats the solve and takes every later one.  ``fallback``
    then says why and from which solve on, and ``fallback_details`` holds
    those report lines.  Any other error stops the run.
    """

    def __init__(
        self, chain, host_solve, statistics, *, tail, poisson_solver, named: bool, leave
    ):
        self.chain = chain
        self.host_solve = host_solve
        self.statistics = statistics
        self.tail = tail
        self.poisson_solver = poisson_solver
        self.named = named
        self.leave = leave
        self.solves = 0
        self.fallback: str | None = None
        self.fallback_details: tuple[tuple[str, str], ...] = ()

    def __call__(self, density, initial_potential=None, **kwargs):
        self.solves += 1
        if self.chain is not None:
            from .Hartree.gpu_memory import allocation_failure

            started = perf_counter()
            try:
                result = self.chain.solve(density, initial_potential, **kwargs)
            except Exception as error:
                if self.named or not allocation_failure(error):
                    raise
                self._leave_chain(error)
            else:
                self.statistics.hartree_solve_calls += 1
                self.statistics.hartree_rhs_seconds += self.chain.rhs_seconds
                self.statistics.hartree_linear_solve_seconds += self.chain.solve_seconds
                self.statistics.hartree_total_seconds += perf_counter() - started
                tail = self.tail()
                return result if tail is None else replace(result, boundary_tail=tail)
        return self.host_solve(density, initial_potential, **kwargs)

    def _leave_chain(self, error) -> None:
        chain, self.chain = self.chain, None
        history = chain.host_history()
        device = chain.device_id
        chain.release()
        del chain
        self.fallback_details = tuple(self.leave())
        adopt = getattr(getattr(self.poisson_solver, "solver", None), "adopt_history", None)
        if callable(adopt):
            adopt(history)
        try:
            from .backends.cupy import require_cupy

            cp, _ = require_cupy()
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001 - the release of cached blocks is a courtesy
            pass
        self.fallback = (
            f"off from Hartree solve {self.solves} (auto: {type(error).__name__}: "
            f"{error}); the boundary on the host and the host interface from then on"
        )


def _planned_sector_device_groups(
    representation_count: int,
    device_ids: tuple[int, ...],
    mpi_context,
    costs=None,
    divisible=None,
    unit_devices=None,
) -> dict[int, tuple[int, ...]]:
    """Return the device group of every sector this process will own.

    This is the assignment ``CuPySymmetrySCFEigensolver`` makes, computed
    without it: the boundary geometry is uploaded while that constructor
    still runs.  An MPI rank gets the groups of its sector context, placed by
    ``costs`` (``Eigensolvers.symmetry.sector_placement_costs``, what the
    constructor passes; ``PARSEC_SECTOR_PLACEMENT``) and by whether a CHEBFF
    first solve divides each basis over the devices of a group
    (``divisible``, ``Eigensolvers.symmetry.sector_bases_divisible``); one
    process spreads all sectors over its devices in turn, or places the units
    of irreducible representations by cost (``unit_devices``,
    ``Eigensolvers.symmetry.unit_device_ids``; ``PARSEC_ORBITAL_IRREPS``).
    """

    if mpi_context is not None:
        planned = getattr(mpi_context, "planned_groups", None)
        if callable(planned):
            if costs is not None and getattr(mpi_context, "accepts_sector_costs", False):
                return planned(representation_count, device_ids, costs=costs, divisible=divisible)
            return planned(representation_count, device_ids)
        from .experimental.mpi_scf import sector_device_groups

        return sector_device_groups(
            representation_count,
            mpi_context.size,
            mpi_context.rank,
            device_ids,
            # The one sector of a run without symmetry, which the ranks share.
            shared=int(representation_count) == 1 and mpi_context.size > 1,
        )
    if unit_devices is not None:
        return {index: (int(unit_devices[index]),) for index in range(representation_count)}
    return {
        index: (device_ids[index % len(device_ids)],)
        for index in range(representation_count)
    }


def _process_placement_refusal(reference) -> str | None:
    """Why one process keeps the cyclic deal of its sectors although it places them by size, or ``None``.

    A hybrid functional's exchange needs every sector basis on one device
    (``Hamiltonian/exchange_device.py``; an MPI rank deals the sectors of a
    hybrid in turn as well), a meta-GGA's kinetic-energy density refuses
    sector groups (``Hamiltonian/meta_gga_device.py``), and the LDA+U
    occupations read the sector bases as single arrays
    (``Hamiltonian/hubbard_device.py``): those keep the placement of before.
    """

    problem = getattr(reference, "input", None)
    scf = getattr(problem, "scf", None)
    if scf is None:
        return None
    if getattr(scf, "is_hybrid", False):
        return "a hybrid functional"
    if getattr(scf, "is_meta_gga", False):
        return "a meta-GGA functional"
    if any(getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values()):
        return "LDA+U"
    return None


_PERIODIC_SECTOR_DEVICES_SWITCH = "PARSEC_PERIODIC_SECTOR_DEVICES"
#: The first bases (FP64, every sector, both spin channels) below which one process keeps the sectors of a periodic
#: cell on its first GPU under ``PARSEC_PERIODIC_SECTOR_DEVICES=auto``.
PERIODIC_ONE_DEVICE_BYTES = 1 << 29


def periodic_sector_devices_setting() -> int | None:
    """``PARSEC_PERIODIC_SECTOR_DEVICES``: where one process with several GPUs spreads the sectors of a periodic cell.

    ``auto`` (the default): a periodic cell whose first sector bases (rows times first state counts, FP64, every
    sector and spin channel) total less than :data:`PERIODIC_ONE_DEVICE_BYTES` keeps all its sectors on the first
    GPU of the process (:func:`_periodic_one_device_note`): its SCF is faster there than spread over four A100, or as
    fast, in every cell measured from 8 to 1728 atoms (first runs about twice as fast: every GPU compiles the kernels
    of its sectors), so the other GPUs idle; diamond with 2744 atoms (1.8 GiB) keeps the four, which win there.  ``all``: the placement by size over every GPU, as before; an integer: that
    many bytes instead.  Returns the bytes (``None`` for ``all``).  A wrong value is refused at setup.
    """

    raw = os.environ.get(_PERIODIC_SECTOR_DEVICES_SWITCH, "auto").strip().lower() or "auto"
    if raw == "auto":
        return PERIODIC_ONE_DEVICE_BYTES
    if raw == "all":
        return None
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{_PERIODIC_SECTOR_DEVICES_SWITCH} must be auto, all or a number of bytes >= 0, not {raw!r}")
    return value


def _periodic_one_device_note(reference, decomposition, device_ids, filter_plan) -> str | None:
    """Why one process keeps all the sectors of a periodic cell on its first GPU, or ``None`` where they spread.

    The first bases of the sectors (:func:`.Eigensolvers.symmetry.sector_placement_costs`: rows times first state
    counts; twice for two spin channels) against ``PARSEC_PERIODIC_SECTOR_DEVICES``
    (:func:`periodic_sector_devices_setting`); a cluster, a single device, one sector (of the identity) or no filter
    plan (no first counts) spread as before.
    """

    limit = periodic_sector_devices_setting()
    problem = getattr(reference, "input", None)
    if limit is None or len(tuple(device_ids)) < 2 or filter_plan is None:
        return None
    if int(decomposition.representation_count) <= 1:
        # one sector (of the identity: PARSEC_CUPY_SINGLE_SECTOR=on, the shared full grid of a spin fallback) is
        # shared by the GPUs on purpose
        return None
    if getattr(problem, "periodic_cell", None) is None:
        return None
    from .Eigensolvers.symmetry import sector_placement_costs

    channels = 2 if getattr(getattr(problem, "scf", None), "spin_polarized", False) else 1
    total = 8.0 * channels * float(sum(sector_placement_costs(decomposition, filter_plan)))
    if total >= float(limit):
        return None
    return (
        f"one GPU: the first bases of the {int(decomposition.representation_count)} sectors total "
        f"{total / 2**20:.1f} MiB, below {limit / 2**20:g} MiB ({_PERIODIC_SECTOR_DEVICES_SWITCH})"
    )


def _process_sector_placement(reference, decomposition, device_ids, filter_plan=None):
    """The device groups of the sectors of one process (``None``: the cyclic deal), whether the first solve
    divides their bases, and why the cyclic deal stays (``None`` where it does not or is asked for).

    ``Eigensolvers.symmetry.process_sector_device_groups`` under ``PARSEC_CUPY_SECTOR_PLACEMENT=size``, with
    the costs and the divisible bases of ``filter_plan`` (the filter plan of ``reference``,
    :func:`_filter_plan`, which the driver hands the sector eigensolver too) and ``Eigensolver: chebff`` as
    the first solve that divides them, as the MPI runner asks of the sectors of one rank.
    """

    from .Eigensolvers.symmetry import (
        process_placement_record,
        process_sector_device_groups,
        process_sector_placement_setting,
    )

    setting = process_sector_placement_setting()
    if setting == "cyclic" or len(tuple(device_ids)) < 2:
        return None, None, None
    refusal = _process_placement_refusal(reference)
    if refusal is not None:
        return None, None, f"cyclic with {refusal} (PARSEC_CUPY_SECTOR_PLACEMENT=size)"
    method = getattr(getattr(getattr(reference, "input", None), "eigensolver", None), "method", "chebff")
    divided = str(method).strip().lower() == "chebff"
    # A periodic cell whose sector bases are small keeps them on one GPU (PARSEC_PERIODIC_SECTOR_DEVICES)
    one_device = _periodic_one_device_note(reference, decomposition, device_ids, filter_plan)
    if one_device is not None:
        first = int(tuple(device_ids)[0])
        groups = {int(sector): (first,) for sector in range(int(decomposition.representation_count))}
        record = process_placement_record(groups, device_ids, int(decomposition.representation_count))
        return groups, divided, f"{record}; {one_device}"
    notes: list[str] = []
    groups = process_sector_device_groups(decomposition, device_ids, filter_plan, divided=divided, notes=notes)
    # The record of the placement with the reason of a pool decision (PARSEC_CUPY_SECTOR_POOL_MIN_BYTES).
    note = None
    if groups and notes:
        record = process_placement_record(groups, device_ids, int(decomposition.representation_count))
        note = f"{record}; {'; '.join(notes)}"
    return groups, divided, note


def _sector_placement_costs(decomposition, filter_plan):
    """The placement costs of the sectors, as the sector eigensolver computes them for its MPI context."""

    from .Eigensolvers.symmetry import sector_placement_costs

    return sector_placement_costs(decomposition, filter_plan)


def _unit_device_ids(decomposition, filter_plan, device_ids):
    """The GPU of every unit of irreducible representations in one process, as the sector eigensolver places them."""

    from .Eigensolvers.symmetry import unit_device_ids

    return unit_device_ids(decomposition, filter_plan, device_ids)


def _sector_bases_divisible(decomposition, filter_plan, device_ids=None, mpi_context=None, costs=None):
    """Which sector bases a CHEBFF first solve divides over a group of devices, as the sector eigensolver asks
    (units of irreducible representations: by the units of this rank, ``planned_owned_units``)."""

    from .Eigensolvers.symmetry import planned_owned_units, sector_bases_divisible

    owned = None if device_ids is None else planned_owned_units(decomposition, mpi_context, device_ids, costs)
    return sector_bases_divisible(decomposition, filter_plan, device_ids, owned=owned)


def _select_hartree_device(
    request: int | str | None,
    device_ids: tuple[int, ...],
    sector_device_groups: dict[int, tuple[int, ...]],
) -> int | None:
    """Choose the CUDA device of the Poisson CG, boundary and resident maps.

    The first device of a process also holds the owner's share of a sector,
    so with these objects it sets the memory peak.  (The sector index and
    expansion maps of the eigensolver do not add to it: they stay on the
    host.)  ``auto`` takes the device with the smallest share of sector
    bases, then one that owns no sector, then the last one listed.  With
    sector groups that is never the first device of a process that has
    several.  ``None`` leaves every object on its default device.
    """

    if request is None:
        return None
    if request != "auto":
        if request not in device_ids:
            raise ValueError(
                "PARSEC_HARTREE_DEVICE is not one of the CUDA devices of "
                "this process"
            )
        return request
    if len(device_ids) < 2:
        return None
    share: dict[int, float] = {}
    owners = set()
    for group in sector_device_groups.values():
        owners.add(group[0])
        for device in group:
            share[device] = share.get(device, 0.0) + 1.0 / len(group)
    return min(
        reversed(device_ids),
        key=lambda device: (share.get(device, 0.0), device in owners),
    )


def _reported_cache_key(key: str | None) -> str:
    """Return the provenance text for a content key that may be absent.

    A cache that is switched off does not hash its key, so the report states
    that instead of a digest.
    """

    return "not computed (cache disabled)" if key is None else key


def _reported_pack_workers(storage: str) -> str:
    """Return the threads that packed the tiles of sector stencils stored as ``storage``.

    ``PARSEC_CUPY_IMPLICIT_PACK_WORKERS`` as the packer reads it.  ``mixed``
    storage holds tiles or none: a setting that is no thread count says
    none, since an operator that is to pack tiles refuses it.
    """

    if "tile" not in storage and storage != "mixed":
        return "no tiles packed"
    try:
        return str(pack_worker_setting())
    except ValueError:
        return "no tiles packed"


#: The filter plan of a preparation not asked for yet (``_prepare_single_point``: the plan is asked once).
_NOT_PLANNED = object()


def _filter_plan(reference) -> FilterPlan | None:
    """Return what the input of ``reference`` fixes of the filter work of its run.

    The states are those the SCF resolves: ``States_Num`` or, without it,
    half the electrons and six more.  The buffer is that of the subspace,
    and the applications are counted from the eigensolver settings and the
    SCF steps the input allows
    (:func:`~.backends.implicit_stencil.counted_filter_applications`).
    ``None`` for a system that does not carry its input, whose sector
    stencils then follow the row limits.
    """

    problem = getattr(reference, "input", None)
    scf = getattr(problem, "scf", None)
    eigensolver = getattr(problem, "eigensolver", None)
    if scf is None or eigensolver is None:
        return None
    states = scf.number_of_states
    if states is None:
        states = int(np.ceil(0.5 * float(reference.electron_count))) + 6
    return FilterPlan(
        states=int(states),
        buffer=int(eigensolver.subspace_buffer),
        applications=counted_filter_applications(eigensolver, scf.max_iterations),
    )


def _apply_hartree_boundary_switches(problem):
    """Apply the environment switches of the Hartree boundary to the input.

    ``PARSEC_HARTREE_BOUNDARY=legacy`` restores PARSEC's boundary values:
    no tolerance and no atomic tail.  ``PARSEC_HARTREE_BOUNDARY_TOLERANCE``
    (a number in Ry, or ``off``) and ``PARSEC_HARTREE_ATOMIC_TAIL``
    (``auto``, ``on`` or ``off``) replace the input keywords
    ``Hartree_Boundary_Tolerance`` and ``Hartree_Atomic_Tail``, and
    ``PARSEC_HARTREE_LPOLE`` replaces ``Solver_Lpole``.  They act before the
    reference is prepared, so the plan of the boundary and the key of the
    resident reference cache see the settings that run.
    """

    mode = os.environ.get("PARSEC_HARTREE_BOUNDARY", "auto").strip().lower()
    if mode not in {"auto", "legacy"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY must be auto or legacy")
    changes: dict[str, object] = {}
    tolerance = os.environ.get("PARSEC_HARTREE_BOUNDARY_TOLERANCE", "").strip().lower()
    if tolerance in {"off", "none"}:
        changes["boundary_tolerance"] = None
    elif tolerance:
        try:
            changes["boundary_tolerance"] = float(tolerance)
        except ValueError as error:
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY_TOLERANCE must be a number in Ry or off"
            ) from error
    tail = os.environ.get("PARSEC_HARTREE_ATOMIC_TAIL", "").strip().lower()
    if tail:
        if tail not in {"auto", "on", "off"}:
            raise ValueError("PARSEC_HARTREE_ATOMIC_TAIL must be auto, on, or off")
        changes["atomic_tail"] = tail
    if mode == "legacy":
        if (
            changes.get("boundary_tolerance") is not None
            or changes.get("atomic_tail") == "on"
        ):
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY=legacy excludes a boundary tolerance "
                "and PARSEC_HARTREE_ATOMIC_TAIL=on"
            )
        changes = {"boundary_tolerance": None, "atomic_tail": "off"}
    order = os.environ.get("PARSEC_HARTREE_LPOLE", "").strip()
    if order:
        try:
            changes["multipole_order"] = int(order)
        except ValueError as error:
            raise ValueError("PARSEC_HARTREE_LPOLE must be an integer") from error
        # The switch is the Solver_Lpole of this run, also for an input an
        # earlier plan had resolved.
        changes["minimum_multipole_order"] = None
    if not changes:
        return problem
    return replace(problem, hartree=replace(problem.hartree, **changes))


def _hartree_boundary_tail(reference, *, on_device: bool, device_id: int | None):
    """Build the static atomic tail the boundary plan of ``reference`` has.

    Returns ``None`` for a plan without one.  ``on_device`` evaluates the
    tail values with the kernel of :mod:`.Hartree.cupy_atomic_tail` on
    ``device_id`` (``None``: the current device of the calling thread);
    otherwise host threads do.  The tail records which, for the report.
    """

    plan = getattr(reference, "hartree_boundary", None)
    if plan is None or not plan.atomic_tail:
        return None
    built = getattr(reference, "hartree_boundary_tail", None)
    if built is not None:
        return built
    from parsec_python.Hartree import valence_point_charges

    from .Hartree.atomic_tail import build_atomic_tail_fast

    tail_values = None
    if on_device:
        from functools import partial

        from .Hartree.cupy_atomic_tail import device_tail_values

        tail_values = partial(device_tail_values, device_id=device_id)
    tail = build_atomic_tail_fast(
        reference.grid,
        *valence_point_charges(
            reference.atoms, reference.pseudopotentials, reference.electron_count
        ),
        plan.order,
        tail_values=tail_values,
    )
    return replace(tail, values_from="device kernel") if on_device else tail


#: How far (bohr) an image of a moved atom may lie from its atom for the ionic sums of the moved structure to run on
#: the orbits (:func:`moved_ionic_sum_setting`): positions averaged over the group by a relaxation step are images to
#: their last bits (about 1e-14 bohr), and the sums at the representatives then differ from those at every point by a
#: relative 1e-11 at most, which the orbit average of the SCF loop takes as well.
MOVED_IMAGE_TOLERANCE = 1.0e-10


def moved_ionic_sum_setting() -> str:
    """``PARSEC_IONIC_MOVED_SUMS``: ``orbits`` (the default since 2026-11-19) or ``full``.

    ``orbits``: the GPU ionic sums of a moved structure (a relaxation step, an MD step) run at one point per orbit of
    the full point group of weighted sectors, or of the group of the sectors, where the images of the moved atoms
    under it lie within :data:`MOVED_IMAGE_TOLERANCE` of their atoms and those of the grid points are exact
    (:func:`.Symmetry.exact_images.orbit_sum_points`; positions averaged over the group by a relaxation differ from
    exact images in their last bits), on every grid point otherwise.  ``full``: every grid point, as before.
    ``PARSEC_IONIC_SUM_GRID=full`` sums every point in any case.
    """

    value = os.environ.get("PARSEC_IONIC_MOVED_SUMS", "orbits").strip().lower() or "orbits"
    if value not in {"orbits", "full"}:
        raise ValueError("PARSEC_IONIC_MOVED_SUMS must be orbits or full")
    return value


def _geometry_builder_hooks(selection, *, ionic_sector_rank: bool, orbits=None, report=None) -> dict:
    """Builders of the ionic fields of moved atoms.

    They are the ones :func:`_prepare_reference_physics` gives the preparation
    of the same backend selection, so that a moved structure is built by the
    kernels that built the first one.  ``orbits`` (``(grid, atoms,
    reduction)``, the moved atoms; ``reduction`` may be a tuple of reductions
    tried in order, :func:`_ionic_sum_candidates`): GPU builders sum at one
    point per orbit of the first reduction whose images of those atoms and of
    the grid points are exact (``PARSEC_IONIC_MOVED_SUMS``,
    :func:`moved_ionic_sum_setting`).
    ``report`` (a dict) receives under ``ionic_sum_grid`` the points those
    sums run on, or why every point.
    """

    if selection.finite_difference_builder != "native":
        return {}
    from .backends.native import _load_native
    from .V_ion import NativeIonicBuilders

    try:
        native_module = _load_native()
    except BackendUnavailableError:
        return {}
    if not hasattr(native_module, "RadialGridEvaluator"):
        return {}
    builders = NativeIonicBuilders()
    if selection.selected == "cupy" and _gpu_ionic_requested():
        from .V_ion.cupy_ionic import CupyIonicBuilders

        builders = CupyIonicBuilders(sector_rank=ionic_sector_rank)
        decision = "full grid (PARSEC_IONIC_MOVED_SUMS=full or PARSEC_IONIC_SUM_GRID=full)"
        if (
            orbits is not None
            and orbits[2] is not None
            and moved_ionic_sum_setting() == "orbits"
            and ionic_sum_grid_setting() == "auto"
        ):
            grid, atoms, candidates = orbits
            try:
                # (a reduction, or a tuple of them tried in order: _ionic_sum_candidates)
                reduction, points, obstacles = _orbit_sum_choice(
                    grid, atoms, candidates, atom_tolerance=MOVED_IMAGE_TOLERANCE
                )
            except Exception as error:
                # As at preparation: a test that cannot be made sums every point.
                points, obstacles = None, [f"the test of the images failed: {type(error).__name__}"]
            if points is not None:
                builders.sum_on_orbits(grid, reduction, points)
                decision = _orbit_sum_wedge_text(reduction, obstacles)
            else:
                decision = f"full grid ({obstacles[0] if obstacles else 'no symmetry reduction'})"
        if report is not None:
            report["ionic_sum_grid"] = decision
    return dict(
        local_ionic_builder=builders.build_local_ionic_potential,
        nonlocal_projector_builder=builders.build_nonlocal_projectors,
        atomic_density_builder=builders.superpose_atomic_density,
    )


def _refresh_boundary_tail(builder, reference, *, on_device: bool, device_id):
    """Give a boundary builder the atomic tail of the atoms of ``reference``.

    Returns the :class:`parsec_python.Hartree.AtomicTail` the Hartree solver
    is to add from now on, ``None`` where the plan has no tail or the builder
    evaluates it per exterior point and keeps it to itself.  ``builder`` may
    be ``None``: the caller then passes the returned tail to its solver.
    """

    plan = getattr(reference, "hartree_boundary", None)
    if plan is None or not plan.atomic_tail:
        return None
    if hasattr(builder, "set_atomic_tail"):
        from parsec_python.Hartree import valence_point_charges

        host_values = {}
        if _hartree_tail_on_host():
            from .Hartree.atomic_tail import host_tail_values

            host_values["tail_values"] = host_tail_values
        builder.set_atomic_tail(
            *valence_point_charges(
                reference.atoms, reference.pseudopotentials, reference.electron_count
            ),
            **host_values,
        )
        return _attached_boundary_tail(builder)
    tail = _hartree_boundary_tail(reference, on_device=on_device, device_id=device_id)
    if builder is not None and tail is not None:
        builder.set_boundary_tail(tail)
    return tail


def _attached_boundary_tail(builder):
    """Return the atomic tail a boundary builder adds, ``None`` without one."""

    from parsec_python.Hartree import AtomicTail

    tail = getattr(builder, "boundary_tail", None)
    return tail if isinstance(tail, AtomicTail) else None


def _hartree_boundary_kernel() -> str:
    """Return the requested kernels of the GPU boundary builder.

    ``PARSEC_HARTREE_BOUNDARY_KERNEL`` is ``auto``, ``full`` or ``wedge``.
    Where the boundary plan is engaged and the boundary is built on a GPU,
    ``auto`` takes the wedge kernels if an axis-reflection group reduces
    the Hartree problem and the same kernels under the identity alone, one
    value per unique exterior point of the full grid, if none does; the
    boundary of the resident chain takes the wedge kernels of such a group
    whether the plan is engaged or not.  Elsewhere, and with ``full``, the
    full-grid kernels of PARSEC's boundary run.
    """

    kernel = os.environ.get("PARSEC_HARTREE_BOUNDARY_KERNEL", "auto").strip().lower()
    if kernel not in {"auto", "full", "wedge"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_KERNEL must be auto, full, or wedge")
    return kernel


def _host_boundary_kernel(symmetric: bool) -> str:
    """The ``hartree_boundary_kernel`` line of a native boundary builder."""

    return "native wedge (C++ orbit table)" if symmetric else "native full grid (C++)"


def _hartree_tail_on_host() -> bool:
    """Whether ``PARSEC_HARTREE_ATOMIC_TAIL_VALUES`` asks for host threads.

    ``auto``, the default, takes the values of the static atomic tail from
    the device kernel wherever the orbital backend is CuPy and from host
    threads elsewhere.  ``host`` takes the host threads there too: the same
    Coulomb sum and series in NumPy, which differ from the kernel's at
    round-off.  The serial control of the MPI runner asks for them, so that
    the kernel is not in both sides of a comparison.
    """

    values = os.environ.get("PARSEC_HARTREE_ATOMIC_TAIL_VALUES", "auto").strip().lower()
    if values not in {"auto", "host"}:
        raise ValueError("PARSEC_HARTREE_ATOMIC_TAIL_VALUES must be auto or host")
    return values == "host"


def _hartree_boundary_check_count() -> int:
    """Return the number of points of ``PARSEC_HARTREE_BOUNDARY_CHECK``, 0 for off."""

    raw = os.environ.get("PARSEC_HARTREE_BOUNDARY_CHECK", "0").strip()
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_HARTREE_BOUNDARY_CHECK must be a number of points"
        ) from error
    if count < 0:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_CHECK cannot be negative")
    return count


def _hartree_boundary_check(system, result) -> tuple[tuple[str, str], ...]:
    """Measure the boundary values of the final density against the direct sum.

    With ``PARSEC_HARTREE_BOUNDARY_CHECK=n`` and a GPU boundary builder, the
    boundary values at ``n`` unique exterior points are compared with the
    direct Coulomb sum of the final density over the whole grid, on the
    Hartree device.  The points are a sample: the innermost ones, those of
    the largest atomic tail, those nearest to an atom and random ones in
    equal shares, so ``hartree_boundary_check_max`` is the maximum over
    that sample; ``n`` at or above the number of points takes them all.
    ``hartree_boundary_check_legacy_max`` is the same error of the
    expansion at ``Solver_Lpole`` without a tail, which is PARSEC's.
    """

    count = _hartree_boundary_check_count()
    if not count:
        return ()
    builder = getattr(system.backend, "native_boundary_builder", None)
    kind = type(builder).__name__
    if kind not in _GPU_BOUNDARY_BUILDERS:
        return (
            (
                "hartree_boundary_check",
                "not run: the boundary values are not built on a GPU",
            ),
        )
    started = perf_counter()
    reference = system.reference
    plan = getattr(reference, "hartree_boundary", None)
    minimum_order = None if plan is None else plan.minimum_order
    density = np.ascontiguousarray(result.density, dtype=np.float64)
    from parsec_python.Hartree import valence_point_charges

    positions, charges = valence_point_charges(
        reference.atoms, reference.pseudopotentials, reference.electron_count
    )
    if kind in {"CuPySymmetryMultipoleBoundaryBuilder", "CuPySubgroupMultipoleBoundaryBuilder"}:
        check = builder.boundary_check(
            density[builder.reduction.representative_rows],
            count,
            minimum_order=minimum_order,
            positions=positions,
        )
    elif kind == "CuPyPointMultipoleBoundaryBuilder":
        check = builder.boundary_check(
            density, count, minimum_order=minimum_order, positions=positions
        )
    else:
        check = builder.boundary_check(
            density,
            count,
            minimum_order=minimum_order,
            tail_charges=(
                (positions, charges)
                if plan is not None and plan.atomic_tail
                else None
            ),
            positions=positions,
        )
    error = check["boundary"] - check["direct"]
    legacy = check["legacy"] - check["direct"]
    total = int(check["total"])
    where = (
        "unique exterior points of the symmetry wedge"
        if kind == "CuPySymmetryMultipoleBoundaryBuilder"
        else "unique exterior points"
    )
    return (
        ("hartree_boundary_check_points", str(error.size)),
        (
            "hartree_boundary_check_sample",
            f"all {total} {where}"
            if error.size == total
            else f"{error.size} of {total} {where}: the innermost, those of "
            "the largest atomic tail, those nearest to an atom and random "
            "ones in equal shares",
        ),
        ("hartree_boundary_check_max", f"{np.abs(error).max():.6e} Ry"),
        ("hartree_boundary_check_rms", f"{np.sqrt(np.mean(error * error)):.6e} Ry"),
        ("hartree_boundary_check_legacy_max", f"{np.abs(legacy).max():.6e} Ry"),
        ("hartree_boundary_check_seconds", f"{perf_counter() - started:.6f}"),
    )


def hartree_boundary_check_seconds(result) -> float:
    """Return the seconds the check after the SCF took, 0 where none ran.

    :func:`run_scf` makes the check before it returns, so a caller that
    times the SCF around that call subtracts these seconds.
    """

    details = dict(getattr(getattr(result, "backend", None), "details", ()))
    return float(details.get("hartree_boundary_check_seconds", 0.0))


def _variational_hartree(reference, solver, *, sector_worker: bool, is_periodic: bool, charges, atoms):
    """``(VariationalHartree or None, reason)`` of ``PARSEC_HARTREE_VARIATIONAL`` for the Hartree solver of a route.

    ``solver`` solves ``A u = rho`` with the CG of the route (``None`` where
    the route has none: a periodic cell, a rank without a Hartree solver);
    ``charges()`` returns the point charges of the atomic tail of the present
    structure, ``atoms()`` its atoms.  The switch was validated by the preparation of ``reference``.
    """

    from parsec_python.Hartree.variational import (
        VariationalHartree,
        ZeroBoundaryDeviceSolver,
        hartree_variational_setting,
        variational_hartree_applies,
    )

    setting = hartree_variational_setting()
    problem = reference.input
    if sector_worker:
        return None, "not used (no Hartree solver on this rank)"
    if is_periodic:
        return None, "not used (periodic cell)"
    if setting == "off":
        return None, "off (PARSEC_HARTREE_VARIATIONAL=off: the potential of the Poisson solve)"
    if setting == "slab":
        return None, "off (PARSEC_HARTREE_VARIATIONAL=slab: a cluster keeps the potential of the Poisson solve)"
    if not variational_hartree_applies(
        getattr(problem, "hartree", None),
        getattr(problem, "grid", None),
        forces_wanted=bool(getattr(problem, "forces_wanted", False)),
        setting=setting,
    ):
        if (
            getattr(getattr(problem, "grid", None), "domain_shape", None) != "sphere"
            or getattr(getattr(problem, "hartree", None), "boundary_method", None) == "direct"
        ):
            return None, "not used (boundary values summed directly)"
        if not getattr(problem, "forces_wanted", False):
            return None, "off (auto: no forces are computed)"
        return None, "off (auto: PARSEC's Hartree boundary, Hartree_Boundary_Tolerance: off, keeps PARSEC's potential)"
    if solver is None:
        return None, "not used (no CG of this route for A u = rho)"
    # The solid-harmonic sums run on the device of a GPU CG, on the host beside a host CG.
    device = getattr(getattr(solver, "backend", None), "device_id", None)
    if isinstance(solver, ZeroBoundaryDeviceSolver):
        device = int(solver.operator.data.device.id)
    where = "host" if device is None else f"device {device}"
    estimator = None
    if setting == "auto":
        # The first solve estimates the force change of V' (Hartree.variational.VariationalHartree._decide).
        from functools import partial

        from parsec_python.Hartree.variational import self_consistency_estimator

        sums = None
        if device is not None:
            from .Forces.local_device import cupy_local_forces

            sums = partial(cupy_local_forces, device_id=device)
        estimator = self_consistency_estimator(
            reference.grid, reference.pseudopotentials, problem.pseudopotentials, sums
        )
    return (
        VariationalHartree(
            reference.grid, problem.hartree, solver, charges=charges, atoms=atoms, device=device, estimator=estimator
        ),
        f"{setting}: the derivative of the Hartree energy"
        + (" where the first solve of a structure estimates a force change from the threshold" if estimator is not None else "")
        + f" (one more CG per solve; solid-harmonic sums and sums of the tail on the {where})",
    )


def _hartree_boundary_details(
    reference, tail, builder=None
) -> tuple[tuple[str, str], ...]:
    """Report the boundary plan of ``reference`` and the tail that was built.

    ``tail`` is the tail in rows; a wedge GPU ``builder`` holds its own, per
    exterior point, and reports its maximum and seconds.  Either one says
    what evaluated its values (``PARSEC_HARTREE_ATOMIC_TAIL_VALUES``): the
    entry is what was built, not what the switch asked for.
    """

    plan = getattr(reference, "hartree_boundary", None)
    if plan is None:
        return ()
    maximum = getattr(tail, "maximum", None)
    seconds = getattr(tail, "seconds", None)
    evaluator = getattr(tail, "values_from", None)
    if maximum is None:
        maximum = getattr(builder, "atomic_tail_maximum", None)
        seconds = getattr(builder, "atomic_tail_seconds", None)
        evaluator = getattr(builder, "atomic_tail_values", None)
    details = (
        (
            "hartree_boundary",
            "PARSEC multipole expansion"
            if plan.legacy
            else "multipole expansion with a-priori control "
            "(PARSEC_HARTREE_BOUNDARY=legacy restores PARSEC's)",
        ),
        (
            "hartree_multipole_order",
            f"{plan.order} (Solver_Lpole {plan.minimum_order})",
        ),
        (
            "hartree_boundary_tolerance",
            "off" if plan.tolerance is None else f"{plan.tolerance:.6e} Ry",
        ),
        (
            "hartree_boundary_estimate",
            f"none ({plan.status})"
            if plan.estimates is None
            else f"{plan.estimate_minimum:.6e} Ry at order {plan.minimum_order}, "
            f"{plan.estimate_order:.6e} Ry at order {plan.order} ({plan.status})",
        ),
        ("hartree_atomic_tail", "applied" if plan.atomic_tail else "not applied"),
        # The estimate runs in the first step of the preparation of the
        # rank, before the grid is built.
        ("hartree_boundary_plan_seconds", f"{plan.seconds:.6f}"),
    )
    if plan.cap_reached:
        details += (
            ("hartree_boundary_cap", "tolerance missed at the largest order"),
        )
    if isinstance(maximum, float) and isinstance(seconds, float):
        details += (
            ("hartree_atomic_tail_max", f"{maximum:.6e} Ry"),
            ("hartree_atomic_tail_seconds", f"{seconds:.6f}"),
        )
    if isinstance(evaluator, str):
        details += (("hartree_atomic_tail_values", evaluator),)
    return details


def _build_native_boundary_builder(
    reference,
    hartree_reduction,
    *,
    symmetry_cache_directory: os.PathLike[str] | str | None,
    symmetry_geometry_cache_info,
    device_id: int | None = None,
    tail_on_device: bool = False,
    boundary_tail=None,
    gpu_kernels: bool = False,
    gpu_linear: bool | None = None,
    resident_wedge: bool = False,
    notes: dict | None = None,
):
    """Construct the reusable isolated-boundary geometry.

    This setup depends only on the already-built real-space grid and exact
    symmetry metadata.  It is deliberately kept separate from the Poisson
    solver so the hybrid path can prepare it on a CPU worker while the main
    thread uploads/builds the independent GPU orbital operators.

    ``device_id`` places a GPU builder on that CUDA device; ``None`` keeps
    the current device of the constructing thread.

    Where the boundary plan of ``reference`` has an atomic tail, it is built
    here too, from the atoms and the grid alone, and attached to the
    builder: ``boundary_tail`` if the caller has it already, else a new one
    whose values come from that device if ``tail_on_device`` and from host
    threads otherwise, or wherever ``PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host``
    asks for them.

    ``gpu_kernels`` says that the orbital backend runs on a GPU.  A native
    builder has no orbit table without a reduction, or where the table of a
    raised order exceeds its storage limit, and then repeats the full-grid
    recurrences in every solve.  An engaged plan takes the GPU kernels
    there instead, and so does any plan where the Poisson CG runs on the
    GPU (``gpu_linear``): the multipoles and the right-hand side are then
    formed on the device that solves, in place of host recurrences over
    the full grid (since 2026-11-07).  ``PARSEC_HARTREE_BOUNDARY_BACKEND=
    native`` or ``PARSEC_HARTREE_BOUNDARY_KERNEL=full`` asks for the former
    route by name.  A builder with an orbit table stays on the host.

    ``gpu_linear`` says that the Poisson CG runs on a GPU, which
    ``PARSEC_HARTREE_BOUNDARY_BACKEND=auto`` requires too; ``None`` reads
    it from ``PARSEC_HARTREE_LINEAR_BACKEND=cupy`` (a caller outside the
    preparation, which has no resolved choice).

    ``resident_wedge`` says that the device-resident chain will solve on
    ``hartree_reduction`` (:func:`_resident_hartree_plan`).  The boundary of
    that wedge is then built on the GPU in place of the orbit table (since
    2026-11-07): the wedge kernels where an axis-reflection group reduces
    the problem, whether the plan is engaged or not, and otherwise the
    kernels of the plan.  A GPU builder that cannot be made leaves the
    orbit table, and ``notes["resident_boundary"]`` says why (``"gpu"``
    where the GPU builder was made for the chain).
    """

    from .Hartree.native_boundary import (
        NativeMultipoleBoundaryBuilder,
        NativeSymmetryMultipoleBoundaryBuilder,
    )

    started = perf_counter()
    tail_on_host = _hartree_tail_on_host()

    def with_tail(builder):
        tail = boundary_tail
        if tail is None:
            tail = _hartree_boundary_tail(
                reference,
                on_device=tail_on_device and not tail_on_host,
                device_id=device_id,
            )
        if tail is not None:
            builder.set_boundary_tail(tail)
        return builder

    kernel = _hartree_boundary_kernel()
    requested = os.environ.get("PARSEC_HARTREE_BOUNDARY_BACKEND", "").strip().lower()
    if requested not in {"", "native", "cupy", "auto"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_BACKEND must be native, cupy, or auto")
    boundary_backend = requested or "native"
    plan = getattr(reference, "hartree_boundary", None)
    engaged = plan is not None and plan.engaged
    order = reference.input.hartree.multipole_order
    if gpu_linear is None:
        gpu_linear = os.environ.get(
            "PARSEC_HARTREE_LINEAR_BACKEND", ""
        ).strip().lower() == "cupy"
    if boundary_backend == "auto":
        from .Hartree.cupy_boundary import auto_gpu_boundary_requested

        wedge_rows = reference.grid.size if hartree_reduction is None else hartree_reduction.wedge_size
        boundary_backend = "cupy" if (
            gpu_linear and auto_gpu_boundary_requested(wedge_rows, order)
        ) else "native"
    # The resident chain takes the boundary of the wedge from the GPU, where
    # the native builder would keep its orbit table on the host; only
    # ``PARSEC_HARTREE_BOUNDARY_BACKEND=native`` or ``..._KERNEL=full`` by
    # name keep it there.
    resident_gpu = (
        resident_wedge
        and boundary_backend != "cupy"
        and gpu_kernels
        and gpu_linear
        and hartree_reduction is not None
        and requested != "native"
        and kernel in {"auto", "wedge"}
    )
    if kernel == "wedge" and boundary_backend != "cupy" and not resident_gpu:
        raise ValueError(
            "PARSEC_HARTREE_BOUNDARY_KERNEL=wedge needs the GPU boundary "
            "(PARSEC_HARTREE_BOUNDARY_BACKEND=cupy)"
        )

    def gpu_builder(wedge_kernels: bool = False):
        """Return a GPU builder and whether it works on the symmetry wedge.

        ``wedge_kernels`` takes the kernels of the wedge wherever an
        axis-reflection group reduces the problem, also for a plan that is
        not engaged (the resident chain).
        """

        from .Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
            CuPyPointMultipoleBoundaryBuilder,
            CuPySubgroupMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
            diagonal_boundary_requested,
            wedge_kernels_supported,
        )

        wedge = wedge_kernels_supported(hartree_reduction)
        # A group with permutations (the orbitals of a tetrahedral or cubic
        # cluster): the wedge kernels on its diagonal subgroup, the fields on
        # its own wedge (PARSEC_HARTREE_DIAGONAL_BOUNDARY=off keeps the
        # kernels under the identity on the full grid).
        diagonal = None
        if (
            not wedge
            and hartree_reduction is not None
            and getattr(hartree_reduction, "operations", None) is not None
            # (a cluster's: no translations)
            and getattr(hartree_reduction, "translations", None) is None
            and diagonal_boundary_requested()
        ):
            from .Symmetry import diagonal_subgroup_reduction

            diagonal = diagonal_subgroup_reduction(reference.grid, hartree_reduction.operations)
        if diagonal is not None:
            wedge = True
        if kernel == "wedge" and not wedge:
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY_KERNEL=wedge needs an axis-reflection "
                "reduction of the Hartree problem"
            )
        if not (
            kernel == "wedge"
            or (kernel == "auto" and (engaged or (wedge_kernels and wedge)))
        ):
            builder = CuPyMultipoleBoundaryBuilder(
                reference.grid, order, device_id=device_id
            )
            return with_tail(builder), False
        if diagonal is not None:
            builder = CuPySubgroupMultipoleBoundaryBuilder(
                reference.grid, hartree_reduction, diagonal, order, device_id=device_id
            )
        elif wedge:
            builder = CuPySymmetryMultipoleBoundaryBuilder(
                reference.grid, hartree_reduction, order, device_id=device_id
            )
        else:
            # No group of axis reflections: the same kernels under the
            # identity alone, with the full-grid interface.
            builder = CuPyPointMultipoleBoundaryBuilder(
                reference.grid, order, device_id=device_id
            )
        if plan is not None and plan.atomic_tail:
            from parsec_python.Hartree import valence_point_charges

            # Evaluated per unique exterior point, on the device of the
            # builder unless host threads are asked for.
            host_values = {}
            if tail_on_host:
                from .Hartree.atomic_tail import host_tail_values

                host_values["tail_values"] = host_tail_values
            builder.set_atomic_tail(
                *valence_point_charges(
                    reference.atoms,
                    reference.pseudopotentials,
                    reference.electron_count,
                ),
                **host_values,
            )
        return builder, wedge

    if boundary_backend == "cupy":
        builder, symmetry_boundary = gpu_builder()
        return builder, symmetry_boundary, None, perf_counter() - started
    if resident_gpu:
        try:
            builder, symmetry_boundary = gpu_builder(wedge_kernels=_RESIDENT_WEDGE_KERNELS)
        except (RuntimeError, MemoryError) as error:
            # As below: the route of a default must not fail where the
            # native one runs.  The orbit table then serves the host
            # interface, and the chain is not built.
            if notes is not None:
                notes["resident_boundary"] = (
                    f"the GPU boundary could not be built ({type(error).__name__}: {error})"
                )
        else:
            if notes is not None:
                notes["resident_boundary"] = "gpu"
            return builder, symmetry_boundary, None, perf_counter() - started
    builder = None
    symmetry_boundary = False
    cache_info = None
    if hartree_reduction is not None:
        try:
            builder = NativeSymmetryMultipoleBoundaryBuilder(
                reference.grid,
                hartree_reduction,
                order,
                cache_directory=symmetry_cache_directory,
                cache_key_seed=(
                    None
                    if symmetry_geometry_cache_info is None
                    else symmetry_geometry_cache_info.key
                ),
            )
        except RuntimeError:
            # No orbit table: it exceeds its storage limit, or a previously
            # installed 0.3 extension has none.  The full grid remains.
            pass
        else:
            symmetry_boundary = True
            cache_info = getattr(builder, "cache_info", None)
    if (
        builder is None
        and gpu_kernels
        and (engaged or gpu_linear)
        and requested != "native"
        and kernel == "auto"
    ):
        try:
            builder, symmetry_boundary = gpu_builder()
        except (RuntimeError, MemoryError):
            # An extension without the geometry exporter of the GPU builders,
            # or a device without the room: the route that was not asked
            # for must not fail where the native one runs.
            builder = None
        else:
            return builder, symmetry_boundary, None, perf_counter() - started
    if builder is None:
        builder = NativeMultipoleBoundaryBuilder(reference.grid, order)
    return (
        with_tail(builder),
        symmetry_boundary,
        cache_info,
        perf_counter() - started,
    )


def _selection_for_problem(
    selection: BackendSelection, problem: SinglePointInput
) -> BackendSelection:
    """Route a periodic problem's static construction to the reference builder.

    The native C++ finite-difference/ionic construction kernels and the
    native/CuPy multipole-boundary Hartree solve are both isolated-cluster
    specific -- neither is periodic-wraparound aware.  The periodic routes
    have their own device paths (``V_ion/periodic_local_device.py`` and
    ``Hartree/periodic_device.py``), chosen inside the reference preparation
    and the periodic Hartree branch, not by these two fields.  Forcing
    ``finite_difference_builder``/``hartree_backend`` to the reference/scipy
    values here means every downstream branch keyed on those two fields
    (native boundary setup, native XC, the diagnostic implementation
    strings) is automatically inert for a periodic problem without needing
    its own ``periodic_cell`` check.

    ``selected`` -- the Hamiltonian-apply execution backend used for the
    eigensolver's repeated matrix-vector products -- is deliberately left
    alone: ``negative_laplacian``/``nonlocal_operator`` are plain sparse
    matrices regardless of periodicity (see the comment in
    :func:`_build_backend`), so CuPy/native can still accelerate that step
    for a periodic problem. Only the Hartree *solve* itself still needs an
    explicit ``periodic_cell`` branch in :func:`prepare_single_point`, since
    forcing ``hartree_backend="scipy"`` would otherwise route into the
    isolated-only scipy Hartree path rather than skipping it.
    """

    if getattr(problem, "periodic_cell", None) is None:
        return selection
    if (
        selection.finite_difference_builder == "reference"
        and selection.hartree_backend == "scipy"
    ):
        return selection
    return replace(
        selection,
        finite_difference_builder="reference",
        hartree_backend="scipy",
        fallback_reasons=selection.fallback_reasons
        + (
            "periodic cell: the cluster construction and multipole-boundary "
            "Hartree kernels (native/CuPy) are not used -- the periodic routes "
            "build the wrapped finite differences on the host and, where a GPU "
            "is present, the local ionic potential and the Hartree solve on "
            "the device (hartree_implementation names the solve)",
        ),
    )


def _resolve_and_prepare_reference(
    problem: SinglePointInput,
    backend: BackendName | str,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
    orbital_operators_only: bool = False,
    ionic_sector_rank: bool = False,
    ionic_field_overlap: _OverlappedIonicFields | None = None,
    preparation: _Preparation | None = None,
):
    """Overlap CUDA driver discovery with independent CPU setup.

    ``cudaGetDeviceCount`` initializes the NVIDIA driver and can consume one
    to three seconds in a fresh Windows process.  The reference grid,
    pseudopotential, ionic, and finite-difference construction does not depend
    on that driver state.  For the production ``auto`` path, execute both
    prerequisites concurrently and join them before symmetry/device
    construction. Explicit backend requests retain their strict sequential
    validation order.

    The provisional choice is used only to select the static finite-difference
    builder.  It is validated against the authoritative backend resolution;
    an availability change or a test double that selects another builder
    triggers an exact rebuild instead of silently composing incompatible
    components.

    ``defer_native_laplacian`` asks for the deferred descriptor in place of
    the full-grid matrix.  Only the CuPy backend can use it, so another
    selected backend gets the matrix, as without the request.

    ``ionic_sector_rank`` is passed on to the GPU ionic sums, which exclude
    the overlap.  The devices they ran on are returned under
    ``ionic_sum_devices``, ``None`` if none ran in this call: sums left to
    ``ionic_field_overlap`` run after it, and their devices are read where
    the overlap is joined.

    ``preparation`` is passed on to the reference preparation, and its
    clock records the backend resolution and the reference preparation.
    """

    from .resources import device_policy

    normalized = str(backend).strip().lower()
    gpu_ionic = _gpu_ionic_requested(periodic=getattr(problem, "periodic_cell", None) is not None, backend=normalized)
    cpu_only = device_policy() == "cpu"
    if gpu_ionic and cpu_only:
        raise ValueError(
            "PARSEC_IONIC_BACKEND=cupy puts the ionic sums on the GPU, but a CPU "
            "run was requested (--device cpu or PARSEC_DEVICE=cpu)"
        )
    ionic_report: dict = {}

    def record(name, started):
        if preparation is not None:
            preparation.stages.add(name, started)

    def prepare_reference(selection, defer):
        options = {}
        if defer:
            options.update(
                defer_native_laplacian=True,
                deferred_laplacian_cache_directory=(
                    deferred_laplacian_cache_directory
                ),
            )
        if orbital_operators_only:
            options.update(orbital_operators_only=True)
        if gpu_ionic:
            options.update(
                ionic_sector_rank=ionic_sector_rank, ionic_report=ionic_report
            )
        if ionic_field_overlap is not None:
            options.update(ionic_field_overlap=ionic_field_overlap)
        if preparation is not None:
            options.update(preparation=preparation)
        started = perf_counter()
        reference = _prepare_reference_physics(problem, selection, **options)
        record("reference_preparation", started)
        return reference

    overlap_requested = os.environ.get(
        "PARSEC_OVERLAP_CUDA_INITIALIZATION", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    # GPU setup must wait for authoritative device selection; the speculative
    # native selection used by the CPU overlap path cannot select it safely.
    can_overlap = overlap_requested and normalized == "auto" and not gpu_ionic
    if not can_overlap:
        resolution_started = perf_counter()
        selection = _selection_for_problem(resolve_backend(backend, problem), problem)
        resolution_seconds = perf_counter() - resolution_started
        record("backend_resolution", resolution_started)
        reference_started = perf_counter()
        reference = prepare_reference(
            selection,
            defer_native_laplacian and selection.selected == "cupy",
        )
        if selection.selected != "cupy":
            # Only the CuPy backend reads a deferred Laplacian (the periodic box defers its matrix whatever the
            # backend: PARSEC_PERIODIC_DEFER_LAPLACIAN): the host backends get the matrix, as on the overlap path.
            reference = _reference_with_matrix(reference)
        reference_seconds = perf_counter() - reference_started
        return selection, reference, {
            "cuda_initialization_overlap": "disabled",
            "backend_resolution_seconds": resolution_seconds,
            "reference_preparation_seconds": reference_seconds,
            "backend_reference_overlapped_seconds": 0.0,
            "ionic_sum_devices": ionic_report.get("devices"),
        }

    # The auto static builder depends only on native availability, not on the
    # CUDA probe running in the worker.  This inexpensive check may be
    # repeated by authoritative resolution after CUDA discovery.
    from .backends.selection import _native_status

    native_available, _ = _native_status()
    provisional = _selection_for_problem(
        BackendSelection(
            requested="auto",
            selected="native" if native_available else "scipy",
            finite_difference_builder=(
                "native" if native_available else "reference"
            ),
            hartree_backend="native" if native_available else "scipy",
        ),
        problem,
    )

    # The probe decides whether CuPy executes.  Defer for it only where CuPy
    # is installed at all, so that a host without it prepares the matrix
    # inside the overlap exactly as before.
    # Nor in a CPU run, where CuPy will not execute whether installed or not.
    defer_provisionally = (
        defer_native_laplacian and _cupy_installed() and not cpu_only
    )
    if preparation is not None:
        # The probe already runs beside this reference preparation, and the
        # backend is not known to it: the symmetry maps stay in line.
        preparation.symmetry_maps = None

    overlap_started = perf_counter()

    def timed_resolution():
        started = perf_counter()
        selected = _selection_for_problem(resolve_backend(backend, problem), problem)
        seconds = perf_counter() - started
        record("backend_resolution", started)
        return selected, seconds

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="parsec-cuda-init",
    ) as executor:
        future = executor.submit(timed_resolution)
        reference_started = perf_counter()
        reference = prepare_reference(provisional, defer_provisionally)
        reference_seconds = perf_counter() - reference_started
        selection, resolution_seconds = future.result()
    overlap_wall_seconds = perf_counter() - overlap_started

    if (
        selection.finite_difference_builder
        != provisional.finite_difference_builder
    ):
        # Availability changed during the probe, or a caller supplied a
        # backend-selection test double.  Correctness takes priority over the
        # discarded speculative setup.
        reference_started = perf_counter()
        reference = prepare_reference(
            selection,
            defer_native_laplacian and selection.selected == "cupy",
        )
        if selection.selected != "cupy":
            reference = _reference_with_matrix(reference)
        reference_seconds += perf_counter() - reference_started
        overlapped_seconds = 0.0
        overlap_status = "rebuild_after_backend_change"
    else:
        overlapped_seconds = max(
            0.0,
            resolution_seconds + reference_seconds - overlap_wall_seconds,
        )
        overlap_status = "cuda_probe_with_cpu_reference_setup"
        if selection.selected != "cupy":
            # CuPy is installed but was not selected.
            reference_started = perf_counter()
            reference = _reference_with_matrix(reference)
            reference_seconds += perf_counter() - reference_started

    return selection, reference, {
        "cuda_initialization_overlap": overlap_status,
        "backend_resolution_seconds": resolution_seconds,
        "reference_preparation_seconds": reference_seconds,
        "backend_reference_overlapped_seconds": overlapped_seconds,
    }


def prepare_single_point(
    problem: SinglePointInput,
    *,
    backend: BackendName | str = "auto",
    symmetry: SymmetryMode | str = "auto",
    symmetry_cache_directory: os.PathLike[str] | str | None = None,
    mpi_context=None,
    spin_identity_sector: bool = False,
    symmetry_fixed_atoms=None,
    symmetry_constraints: tuple = (),
    sector_forces: bool = False,
    minimum_sector_order: int = 0,
) -> AcceleratedPreparedSinglePointSystem:
    """Prepare physics, automatically reducing exact symmetry when usable.

    ``auto`` (the default) uses every exactly detected Cartesian reflection
    that the selected backend supports and otherwise keeps the full-grid
    algorithm.  ``on`` turns an unusable/nontrivial symmetry into an error;
    ``off`` skips detection and is the reproducible full-grid comparison path.

    ``sector_forces``: the caller averages the forces and moved positions of
    a periodic cell over the group of its sectors (the command line and the
    MPI runner under ``PARSEC_PERIODIC_SECTOR_FORCES=auto``), so the cell
    keeps its sectors below the size floor of ``PARSEC_PERIODIC_SECTOR_FLOOR``
    too: the full grid would leave its forces and moves unaveraged.

    ``symmetry_cache_directory`` names a persistent symmetry cache.  Without
    one, the default here and of the command-line entry points, nothing is
    read or written, which is the fastest first calculation of a structure;
    with one, a process loads or builds the stencils of all sectors.

    With an ``mpi_context`` every rank prepares the symmetry sectors it owns.
    Only the root rank also prepares what the SCF loop itself needs (ionic
    fields, densities, Hartree and XC); on the other ranks those hooks raise.

    ``symmetry_constraints`` are points the operations of the detected group
    of a cluster must also carry onto points of the same label (pseudo-atoms
    that are no part of the physics): the group of the atoms that keeps an
    electric field along an axis (``Polarizability``,
    :mod:`.SCF.symmetry_off_inputs`).  The point charges of the problem join
    them where ``PARSEC_POINT_CHARGE_SYMMETRY=auto``.

    Stages that are independent run on threads beside the calling one.
    Every one of them has ended when this function returns or raises.
    ``preparation_stages`` of the returned system says when each stage of
    every thread began and ended.

    ``spin_identity_sector`` (a spin-polarized ``problem`` on CuPy, with
    ``symmetry='off'``): both channels as one sector of the identity, whose
    basis the GPUs of the process share, or the ranks of ``mpi_context``,
    without a symmetry guard: the full grid that :class:`_SpinSectorFallback`
    gives a run whose guard fired under ``PARSEC_SPIN_FALLBACK_SHARED``.
    ``symmetry_fixed_atoms`` (one flag per atom, ``True`` for an atom that
    ``Move_Flag`` fixes; :func:`parsec_python.Relaxation.run.symmetry_fixed_atoms`)
    has the detection take the fixed atoms as a species of their own, so
    that it finds the subgroup a relaxation with those flags keeps
    (``PARSEC_MOVE_FLAG_SUBGROUP``).  ``None``, the default, detects the
    group of the atoms as they are.

    ``minimum_sector_order`` (a periodic cell, ``symmetry`` other than
    ``on``): sectors whose group has fewer operations are dropped after the
    detection, before they are built, and the cell takes the full grid (the
    molecular dynamics and manual movements of a cell, and the relaxations
    with fixed atoms of the Fortran-derived routines, under
    ``PARSEC_MOVING_SECTORS=auto``, :mod:`.SCF.moving_sectors`).  ``0``, the
    default, keeps every group.
    """

    if spin_identity_sector and not getattr(getattr(problem, "scf", None), "spin_polarized", False):
        raise ValueError("spin_identity_sector is the full grid of the two channels of a spin-polarized run")
    if spin_identity_sector and _normalize_symmetry_mode(symmetry) != "off":
        raise ValueError("spin_identity_sector takes no point group: prepare it with symmetry='off'")
    if getattr(getattr(problem, "scf", None), "is_libxc", False):
        _refuse_libxc_routes(problem, mpi_context=mpi_context)
    if getattr(getattr(problem, "scf", None), "is_hybrid", False):
        _refuse_hybrid_routes(problem, backend=backend, mpi_context=mpi_context)
    if getattr(getattr(problem, "scf", None), "is_meta_gga", False):
        _refuse_meta_gga_routes(problem, mpi_context=mpi_context)
    # PARSEC_CUPY_COMPILE_AHEAD: the kernels the input will need, compiled on background threads beside the
    # preparation (the same binaries; nothing where the command line has handed them over already)
    _start_compile_ahead(problem, backend)
    preparation = _Preparation()
    try:
        return _prepare_single_point(
            problem,
            backend=backend,
            symmetry=symmetry,
            symmetry_cache_directory=symmetry_cache_directory,
            mpi_context=mpi_context,
            preparation=preparation,
            spin_identity_sector=spin_identity_sector,
            symmetry_fixed_atoms=symmetry_fixed_atoms,
            symmetry_constraints=tuple(symmetry_constraints),
            sector_forces=bool(sector_forces),
            **({"minimum_sector_order": int(minimum_sector_order)} if minimum_sector_order else {}),
        )
    finally:
        # A preparation that returns has joined its threads where it read
        # their results; one that raises may have left some running.
        preparation.close()



def _refuse_libxc_routes(problem, *, mpi_context) -> None:
    """Refuse a libxc functional on the MPI routes (``V_xc/libxc.py``).

    No libxc functional has been run on several ranks: neither an LDA or GGA
    on the distributed solvers (``main.py --mpi`` refuses those by name) nor a
    hybrid in the MPI runner of several nodes (``scripts/run_multi_gpu_scf.sh``,
    :mod:`.SCF.hybrid_mpi`), whose root evaluates the semilocal part.  One
    process on the GPUs of one node runs them (``--serial-control``).
    """

    if mpi_context is not None:
        raise NotImplementedError(
            f"the libxc functional {problem.scf.xc_functional!r} is not implemented for the MPI SCF of several "
            "ranks or nodes (not tested there); run it in one process on one node (--serial-control), which "
            "uses all GPUs of that node"
        )


def _refuse_hybrid_routes(problem, *, backend, mpi_context) -> None:
    """Refuse the routes of a hybrid functional that the device exchange does not cover.

    The device Fock loop (SCF/hybrid.py) runs isolated systems in one
    process on the GPUs of one node, unpolarized and spin-polarized, on the
    full grid or in the symmetry sectors with one device per sector basis;
    on several MPI ranks (SCF/hybrid_mpi.py) in the symmetry sectors, with
    a sector context made for the exchange (one device per sector basis,
    the pair tiles on every device of every rank; a periodic cell's sector
    orbitals gathered on every rank as one process gathers them,
    PARSEC_MPI_EXCHANGE_POOL); and periodic cells (bulk
    and slab) at the Gamma point on the full grid of one process, their pair
    densities by FFT (Hamiltonian/bloch_exchange_device.py).
    """

    label = problem.scf.xc_functional
    if mpi_context is not None and not getattr(mpi_context, "exchange", False):
        raise NotImplementedError(
            f"the self-consistent hybrid functional {label!r} needs an MPI sector context made "
            "for the exchange (MPISectorContext(comm, exchange=True), as the MPI runner makes it)"
        )
    if str(backend).lower() not in ("auto", "cupy"):
        raise NotImplementedError(
            f"the self-consistent hybrid functional {label!r} runs on the CuPy backend of the "
            "accelerated path; use --backend auto or cupy, or the reference solver "
            "(reference_main.py)"
        )
    settings = getattr(problem.scf, "exact_exchange", None)
    periodic = getattr(problem, "periodic_cell", None) is not None
    if periodic and getattr(problem.grid, "periodicity", 3) == 1:
        raise NotImplementedError(
            f"the self-consistent hybrid functional {label!r} is not implemented for a wire "
            "(Boundary_Conditions: wire); bulk cells and slabs are"
        )
    if not periodic and getattr(settings, "pair_solver", "kronecker") != "kronecker":
        raise NotImplementedError(
            "the device exchange operator solves the pair densities with the Kronecker "
            "solver only (Exx_Pair_Solver: kronecker); Exx_Pair_Solver: cg runs on the "
            "reference solver (reference_main.py)"
        )
    for name, refused, reason in (
        (
            "PARSEC_CUPY_DISTRIBUTED_FILTER",
            {"1"},
            "a filter spread over several devices does not hold the exchange operator",
        ),
        (
            "PARSEC_CUPY_RITZ_HANDOVER",
            {"triangular"},
            "the exchange operator is built from Ritz vectors",
        ),
        (
            "PARSEC_CUPY_MIXED_FILTER",
            {"on"},
            "the FP32 filter does not hold the exchange operator",
        ),
    ):
        if os.environ.get(name, "").strip().lower() in refused:
            raise NotImplementedError(
                f"{name}={os.environ[name]} is not implemented with the self-consistent hybrid "
                f"functional {label!r}: {reason}"
            )


def _refuse_meta_gga_routes(problem, *, mpi_context) -> None:
    """Refuse the routes of a meta-GGA that its device term does not cover (Hamiltonian/meta_gga_device.py).

    The term runs in one process on the GPUs of one node (full grid or the
    symmetry sectors with one device per sector basis) and on the CPU
    backends.  Filters that bypass the term (captured graphs, a filter
    spread over devices) give way to the ordinary filter by themselves.
    """

    label = problem.scf.xc_functional
    if mpi_context is not None:
        raise NotImplementedError(
            f"the meta-GGA {label!r} is not implemented for the MPI SCF of several ranks or nodes; "
            "run it in one process on one node (--serial-control), which uses all GPUs of that node"
        )
    for name, refused, reason in (
        ("PARSEC_CUPY_MIXED_FILTER", {"on"}, "the FP32 filter does not hold the orbital term -div(vtau grad)"),
        ("PARSEC_CUPY_RITZ_HANDOVER", {"triangular"}, "tau is formed from Ritz vectors"),
    ):
        if os.environ.get(name, "").strip().lower() in refused:
            raise NotImplementedError(
                f"{name}={os.environ[name]} is not implemented with the meta-GGA {label!r}: {reason}"
            )


def _hybrid_xc_evaluators(reference, *, backend, reducer, device_id, core_density, nuclear_gradient=None):
    """``(unpolarized, spin, (start unpolarized, start spin))`` evaluators of a hybrid.

    The semilocal part of the hybrid on the device where
    :func:`.V_xc.device_hybrid_xc.device_hybrid_supported` allows it, and
    the start functional of the Fock loop where ``PARSEC_CUPY_XC`` puts it on
    the device (:func:`.V_xc.accelerated_xc_evaluators`); ``None`` entries
    keep the host functionals of the reference system.  ``nuclear_gradient`` returns the correction of the density
    gradient of the present structure (``PARSEC_GGA_NUCLEAR_GRADIENT``; default: that of ``reference``).
    """

    from types import SimpleNamespace

    from .V_xc.device_xc import nuclear_source

    if nuclear_gradient is None:
        nuclear_gradient = nuclear_source(lambda: reference)

    from parsec_python.SCF.hybrid import hybrid_definition, hybrid_functional_of

    from .V_xc import accelerated_xc_evaluators, device_xc_mode

    scf = reference.input.scf
    spin = bool(getattr(scf, "spin_polarized", False))
    unpolarized = spin_evaluator = None
    if backend == "cupy" and device_xc_mode() != "off":
        from .V_xc.device_hybrid_xc import (
            DeviceHybridXCEvaluator,
            DeviceSpinHybridXCEvaluator,
            device_hybrid_supported,
        )

        functional = hybrid_functional_of(scf)
        from .V_xc.device_libxc import device_libxc_supported

        if device_libxc_supported(functional.semilocal, reference.grid):
            # A libxc hybrid: its semilocal part by libxc's CUDA build (V_xc/device_libxc.py).
            from .V_xc.device_libxc import DeviceLibxcXCEvaluator, DeviceSpinLibxcXCEvaluator

            keywords = dict(
                name=functional.name,
                core_density=core_density,
                reducer=reducer,
                device_id=device_id,
                nuclear_gradient=nuclear_gradient,
            )
            if spin:
                spin_evaluator = DeviceSpinLibxcXCEvaluator(functional.semilocal, reference.grid, **keywords)
            else:
                unpolarized = DeviceLibxcXCEvaluator(functional.semilocal, reference.grid, **keywords)
        elif device_hybrid_supported(functional, reference.grid):
            if spin:
                # On the orbits where the channels are solved in the sectors
                # (``reducer``; None on the full grid).
                spin_evaluator = DeviceSpinHybridXCEvaluator(
                    functional,
                    reference.grid,
                    core_density=core_density,
                    reducer=reducer,
                    device_id=device_id,
                    nuclear_gradient=nuclear_gradient,
                )
            else:
                unpolarized = DeviceHybridXCEvaluator(
                    functional,
                    reference.grid,
                    core_density=core_density,
                    reducer=reducer,
                    device_id=device_id,
                    nuclear_gradient=nuclear_gradient,
                )
    # Zero_Average_Vxc of a 3-D cell (potshift): the device evaluators of the hybrid and of its start functional
    # shift their potentials as the reference system's evaluate_xc does (the default of an unpolarized cell).
    shift = getattr(reference, "_zero_average", None)
    if callable(shift) and shift(None) is True:
        from .V_xc.device_xc import ZeroAverageXC

        unpolarized = None if unpolarized is None else ZeroAverageXC(unpolarized)
        spin_evaluator = None if spin_evaluator is None else ZeroAverageXC(spin_evaluator)
    start_scf = replace(
        scf, xc_functional=hybrid_definition(scf).start_functional, exact_exchange=None
    )
    start_reference = SimpleNamespace(
        input=replace(reference.input, scf=start_scf),
        grid=reference.grid,
        core_density=reference.core_density,
        _zero_average=shift,
    )
    start = accelerated_xc_evaluators(
        start_reference,
        backend=backend,
        reducer=reducer,
        device_id=device_id,
        core_density=core_density,
        nuclear_gradient=nuclear_gradient,
    )
    return unpolarized, spin_evaluator, start


def _count_hybrid_device_bytes(sectors, exchange_device, reference, xc_evaluator, start_xc) -> None:
    """Hand the device bytes of a hybrid to the memory rules of the sectors (``resident_device_bytes``)."""

    from parsec_python.SCF.hybrid import hybrid_definition
    from parsec_python.SCF.single_point import _number_of_states

    counts = dict(sectors.resident_device_bytes)
    for evaluator in (xc_evaluator, *start_xc):
        planned = getattr(evaluator, "planned_device_bytes", None)
        device = getattr(evaluator, "device_id", None)
        if planned is not None and device is not None:
            counts[int(device)] = counts.get(int(device), 0) + int(planned)
    planned = exchange_device.planned_device_bytes(
        reference, hybrid_definition(reference.input.scf), _number_of_states(reference)
    )
    for device, count in planned.items():
        counts[int(device)] = counts.get(int(device), 0) + int(count)
    exchange_device.planned_bytes = planned
    sectors.resident_device_bytes = counts


def _start_xc_with_reducer(start_xc, reference_of, reducer, *, spin_sectors: bool = False):
    """The start evaluators of a hybrid's Fock loop, the host ones on the wedge too.

    A missing unpolarized device evaluator of a symmetry-reduced loop is the
    reference start functional on the full grid with the wedge fields of the
    loop, as the driver gives every host functional; ``None`` elsewhere (the
    reference system's own functional).  So is a missing spin evaluator where
    the spin channels are solved in the sectors (``spin_sectors``).
    ``reference_of()`` returns the reference system of the present structure:
    an ionic move replaces it, and with it the frozen core density the start
    functional reads.
    """

    unpolarized, spin = start_xc
    if reducer is None or (unpolarized is not None and (spin is not None or not spin_sectors)):
        return (unpolarized, spin)
    from parsec_python.SCF.hybrid import hybrid_definition
    from parsec_python.V_xc import XCResult

    def start_reference():
        reference = reference_of()
        scf = reference.input.scf
        start_scf = replace(
            scf, xc_functional=hybrid_definition(scf).start_functional, exact_exchange=None
        )
        return replace(reference, input=replace(reference.input, scf=start_scf))

    def start_xc_evaluator(density):
        evaluated = start_reference().evaluate_xc(reducer.to_full(density))
        return XCResult(
            potential=reducer.from_full(evaluated.potential),
            energy_per_electron=reducer.from_full(evaluated.energy_per_electron),
            energy_density=reducer.from_full(evaluated.energy_density),
            total_energy=evaluated.total_energy,
        )

    scf = getattr(getattr(reference_of(), "input", None), "scf", None)
    if spin is None and spin_sectors and getattr(scf, "is_hybrid", False):
        # The spin channels in the sectors: the host start functional on
        # their orbit fields (SCF/spin_sectors.py).
        from .SCF.spin_sectors import HostSpinXCOnOrbits

        spin = HostSpinXCOnOrbits(
            lambda up, down: start_reference().evaluate_spin_xc(up, down),
            reducer,
            _readable_spin_xc_label(hybrid_definition(scf).start_functional),
        )
    return (start_xc_evaluator if unpolarized is None else unpolarized, spin)


def _prepare_single_point(
    problem: SinglePointInput,
    *,
    backend: BackendName | str,
    symmetry: SymmetryMode | str,
    symmetry_cache_directory: os.PathLike[str] | str | None,
    mpi_context,
    preparation: _Preparation,
    spin_identity_sector: bool = False,
    symmetry_fixed_atoms=None,
    symmetry_constraints: tuple = (),
    sector_forces: bool = False,
    minimum_sector_order: int = 0,
) -> AcceleratedPreparedSinglePointSystem:
    """Run the stages of :func:`prepare_single_point`, which joins their threads."""

    stages = preparation.stages
    is_periodic = getattr(problem, "periodic_cell", None) is not None
    # The input as it was given: a spin-polarized run that leaves the
    # symmetry sectors is prepared again from it (_spin_sector_fallback).
    given_problem = problem
    # Two spin channels are solved on the full grid unless
    # Spin_Symmetry_Sectors puts them in the representation sectors
    # (_spin_sector_refusal); the MPI ranks that divide the sectors keep one
    # subspace per representation and serve no spin channels.
    spin_polarized = bool(
        getattr(getattr(problem, "scf", None), "spin_polarized", False)
    )
    # PARSEC_SPIN_FLIP_SYMMETRY: read (and refused if unknown) before anything is built.
    from .SCF.spin_flip import spin_flip_setting, spin_flip_stability_setting

    spin_flip_mode = spin_flip_setting()
    # (and PARSEC_SPIN_FLIP_STABILITY, the test of the states of that route, read so that a value it does not know is
    # refused)
    spin_flip_stability = spin_flip_stability_setting()
    spin_sector_refusal = _spin_sector_refusal(problem, mpi_context=mpi_context) if spin_polarized else None
    if spin_polarized and mpi_context is not None and spin_sector_refusal is not None:
        # The ranks divide the symmetry sectors; two channels on the full grid
        # run in one process.
        raise NotImplementedError(
            "a spin-polarized calculation whose channels stay on the full grid "
            f"({spin_sector_refusal}) cannot run as an MPI SCF: the ranks divide the "
            "symmetry sectors. Run it in one process (--serial-control or main.py)"
        )
    _refuse_corrections(problem, is_periodic=is_periodic, mpi_context=mpi_context)
    point_charges = bool(getattr(problem, "point_charges", ()))
    if mpi_context is not None and (
        point_charges
        or getattr(getattr(problem, "scf", None), "symmetrize_density", False)
    ):
        # The ranks of the MPI runner have not run these: refused, not ignored.
        raise NotImplementedError(
            "point charges (Add_Point_Charges) and Symmetrize_Charge_Density do "
            "not run as an MPI SCF; run the input in one process (main.py)"
        )
    from .SCF.symmetry_off_inputs import point_charge_symmetry_setting

    # (read in every run, so that a value it does not know is refused)
    point_charge_group = point_charge_symmetry_setting() == "auto" and not is_periodic
    if is_periodic:
        # PARSEC_CONFINED_SETUP, PARSEC_SPLINE_SETUP and PARSEC_PERIODIC_OBLIQUE_SPLIT: refused here if unknown, before
        # the reference preparation reads them
        from .V_ion.confined_setup import confined_setup_setting
        from .V_ion.spline_setup import spline_setup_setting

        confined_setup_setting()
        spline_setup_setting()
        periodic_oblique_split()
    if point_charges and not point_charge_group:
        # The potential of the point charges is in the ionic potential of the
        # reference preparation (ionpot.f90), which has no point group of the
        # atoms and charges: the sectors of the atoms alone would average it.
        # (PARSEC_POINT_CHARGE_SYMMETRY=auto: the group of the atoms that
        # carries every charge onto a charge of its value, below.)
        if _normalize_symmetry_mode(symmetry) == "on":
            raise ValueError(
                "symmetry='on' cannot be used with point charges (Add_Point_Charges): "
                "the calculation uses no point group"
                + ("" if is_periodic else " (PARSEC_POINT_CHARGE_SYMMETRY=off)")
            )
        symmetry = "off"
    if getattr(problem, "embedding", None) is not None:
        # FDET of PARSEC 1.5 (Potential_Field): the frozen sample field has no
        # point group of the tip atoms, and only this process holds it.
        if mpi_context is not None:
            raise NotImplementedError(
                "FDET (Potential_Field) does not run as an MPI SCF; run the input in one process (main.py)"
            )
        if _normalize_symmetry_mode(symmetry) == "on":
            raise ValueError(
                "symmetry='on' cannot be used with FDET (Potential_Field): the frozen sample field "
                "breaks the point group of the tip"
            )
        symmetry = "off"
    # A periodic cell has no Hartree boundary values.  The switches are read
    # for it too, so that a value they do not know is refused in every run,
    # and leave its input as it is.
    boundary_switched = _apply_hartree_boundary_switches(problem)
    if not is_periodic:
        problem = boundary_switched
    # Read here so that a value they do not know is refused in every run.
    from .SCF.density_symmetry import symmetrized_density_sectors_setting

    symmetrized_density_sectors_setting()
    _hartree_boundary_kernel()
    _hartree_boundary_check_count()
    ionic_sum_grid_setting()
    from .Forces.local_device import local_force_sums_setting

    # (PARSEC_LOCAL_FORCE_SUMS, the local forces of a cluster in sectors, read only by the forces after the SCF)
    local_force_sums_setting()
    tail_on_host = _hartree_tail_on_host()
    ionic_fields = _ionic_field_overlap(mpi_context, periodic=is_periodic, backend=backend)
    if ionic_fields is not None:
        preparation.join_on_exit(ionic_fields.close)
    symmetry_mode = _normalize_symmetry_mode(symmetry)
    from .Symmetry import periodic_symmetry_setting
    from .Symmetry.axis_reflection import (
        orbital_irreps_setting,
        periodic_abelian_translation_setting,
        periodic_centre_setting,
        periodic_defect_centre_setting,
        periodic_trivial_origin_setting,
        sector_orbits_exact_setting,
        symmetry_atom_tolerance,
    )

    # Read in every run, so that a value it does not know is refused.
    periodic_symmetry_mode = periodic_symmetry_setting()
    # (PARSEC_ORBITAL_IRREP_ROWS, which the units of PARSEC_ORBITAL_IRREPS read)
    from .Symmetry.axis_reflection import orbital_irrep_rows_setting

    orbital_irrep_rows_setting()
    # (and so are the switches of the detection: PARSEC_SYMMETRY_ATOM_TOLERANCE, PARSEC_PERIODIC_TRIVIAL_ORIGIN,
    # PARSEC_PERIODIC_SYMMETRY_CENTRE, PARSEC_PERIODIC_DEFECT_CENTRE, PARSEC_PERIODIC_ABELIAN_TRANSLATIONS,
    # PARSEC_ORBITAL_IRREPS)
    atom_tolerance = symmetry_atom_tolerance()
    # (and PARSEC_SECTOR_ORBITS_EXACT, whether a structure that keeps the group only within a loose tolerance takes
    # the projectors of one atom per orbit)
    sector_orbits_exact_setting()
    trivial_origin_mode = periodic_trivial_origin_setting()
    periodic_centre_setting()
    periodic_defect_centre_setting()
    periodic_abelian_translation_setting()
    # (and the switches of the local forces of a cell, which are used after the first SCF:
    # PARSEC_PERIODIC_LOCAL_ORBITS, PARSEC_PERIODIC_LOCAL_BALLS and their siblings)
    from .Forces.periodic_local import validate_local_force_settings

    validate_local_force_settings()
    from .SCF.kpoint_sectors import kpoint_sector_tolerance_setting

    # (and PARSEC_KPOINT_SECTOR_TOLERANCE, the tolerance of the cell group of the k-point sectors: the 1x1x1 driver)
    kpoint_sector_tolerance_setting()
    orbital_irreps_setting()
    from .SCF.gamma_spin import gamma_spin_sector_setting

    # (and PARSEC_GAMMA_SPIN_SECTORS, the route of a spin-polarized cell: SCF/gamma_spin.py, with the operations of
    # test 2 of its guard, PARSEC_SPIN_GUARD_CELL_GROUP: SCF/spin_sectors.py)
    gamma_spin_sector_setting()
    from .SCF.spin_sectors import (
        spin_fallback_shared_setting,
        spin_guard_cell_group_setting,
        spin_guard_doublets_setting,
    )

    spin_guard_cell_group_setting()
    # (and PARSEC_SPIN_GUARD_DOUBLETS, the doublets that one weighted sector holds)
    spin_guard_doublets_setting()
    # (and PARSEC_SPIN_FALLBACK_SHARED, the layout of the full grid of a spin run whose guard fires)
    spin_fallback_shared_setting()
    from .Forces.sector_symmetry import fortran_sector_forces_setting

    # (and PARSEC_FORTRAN_SECTOR_FORCES, which compute_forces and move_atoms read for a cell of the Fortran-derived
    # routines in sectors: read here so that a value it does not know is refused before the SCF of a library caller)
    fortran_sector_forces_setting()
    from .SCF.gamma_features import gamma_feature_settings

    # (and the switches of TS, LDA+U, meta-GGA and libxc in the Gamma sectors: SCF/gamma_features.py)
    gamma_feature_settings()
    from .SCF.gamma_hybrid_route import gamma_hybrid_sector_forces_setting

    # (and PARSEC_GAMMA_HYBRID_SECTOR_FORCES, whether a small Gamma hybrid with forces or moves averaged over the
    # group averages them on the full grid or takes the sectors: _gamma_hybrid_in_sectors and
    # _gamma_hybrid_full_grid_forces)
    gamma_hybrid_sector_forces_setting()
    hybrid_full_grid_forces = is_periodic and _gamma_hybrid_full_grid_forces(
        problem,
        mpi_context,
        sector_forces=bool(sector_forces),
        symmetry_mode=symmetry_mode,
        periodic_symmetry_mode=periodic_symmetry_mode,
    )
    from .Hamiltonian.hubbard_device import ldau_projector_orbits_setting
    from .SCF.gamma_features import (
        ldau_sector_forces_setting,
        ldau_spin_full_grid_setting,
        ldau_spin_sector_setting,
    )

    # (and the projector orbits of LDA+U in the sectors, PARSEC_LDAU_PROJECTOR_ORBITS, its spin sectors,
    # PARSEC_LDAU_SPIN_SECTORS, the full grid of its small spin cells, PARSEC_LDAU_SPIN_FULL_GRID, and the sector
    # forces of its cells, PARSEC_LDAU_SECTOR_FORCES, which the command line reads: read in every run, so that a
    # value they do not know is refused before the SCF)
    ldau_orbits = ldau_projector_orbits_setting()
    ldau_spin_sector_setting()
    ldau_spin_full_grid_setting()
    ldau_sector_forces_setting()
    from .Hamiltonian.hubbard_device import ldau_weighted_sectors_setting

    # (PARSEC_LDAU_WEIGHTED_SECTORS too, which _orbital_group_refusal reads for an LDA+U run only)
    ldau_weighted_sectors_setting()
    from .Symmetry.axis_reflection import slab_wire_sectors_setting
    from .Symmetry.oblique_frame import oblique_sectors_setting

    # (PARSEC_SLAB_WIRE_SECTORS, PARSEC_OBLIQUE_SECTORS and PARSEC_OBLIQUE_FULL_GROUP, the sectors of a wire, slab or
    # oblique cell: read in every run, so that a value they do not know is refused before the SCF)
    from .Symmetry.oblique_frame import oblique_full_group_setting

    slab_wire_sectors_setting()
    oblique_sectors_setting()
    oblique_full_group_setting()
    # A periodic cell in an MPI sector context: one sector of the identity
    # whose basis the ranks share (a bulk cell at Gamma; see
    # periodic_sector_refusal for what that route does not take).
    # An unpolarized Gamma-point hybrid on the ranks of the MPI runner: its sectors or the one sector of the identity
    # (_mpi_periodic_hybrid, since 2026-11-22; None: not such a run).
    mpi_hybrid = _mpi_periodic_hybrid(problem, mpi_context) if is_periodic else None
    periodic_refusal = (
        periodic_sector_refusal(
            problem,
            hybrid_sectors=(
                _gamma_hybrid_in_sectors(problem, mpi_context, sector_forces=sector_forces) or mpi_hybrid is not None
            ),
            features=mpi_context is None,
        )
        if is_periodic
        else None
    )
    # The cell that route takes may use its point group about the origin of
    # the cell: the sectors of that group instead of the one of the identity
    # (PARSEC_PERIODIC_SYMMETRY=off keeps the identity).
    periodic_symmetry = (
        is_periodic
        and periodic_refusal is None
        and symmetry_mode != "off"
        and periodic_symmetry_mode == "auto"
    )
    # (read in every run, so that a value it does not know is refused)
    periodic_floor = periodic_sector_floor_setting()
    # (and PARSEC_PERIODIC_SPIN_SECTOR_FLOOR, the floor of a spin-polarized cell)
    periodic_spin_floor = periodic_spin_sector_floor_setting()
    from .Symmetry.axis_reflection import sector_overhead_entries

    # (and PARSEC_SECTOR_OVERHEAD, the fixed cost of a sector in the choice of the weighted plan of a cell)
    sector_overhead_entries()
    from .SCF.band_sectors import band_kpoint_sectors_setting
    from .Symmetry.sector_shares import weighted_first_share_setting

    weighted_first_share_setting()
    # (PARSEC_BAND_KPOINT_SECTORS, read by the band stage after the SCF)
    band_kpoint_sectors_setting()
    # An unpolarized cell whose basis is below the floor keeps the full grid in one process
    # (PARSEC_PERIODIC_SECTOR_FLOOR; --symmetry on keeps the sectors; a hybrid has the same floor in
    # _gamma_hybrid_in_sectors, below which its forces and moves are averaged on the full grid:
    # _gamma_hybrid_full_grid_forces).  A spin-polarized cell with a native LDA or GGA has a floor of its own
    # (PARSEC_PERIODIC_SPIN_SECTOR_FLOOR).  Forces and moves averaged over the group of the sectors (sector_forces)
    # keep the sectors too: the full grid would not average them; but not those of a cell whose sectors are those of
    # the frame of its grid indices (PARSEC_OBLIQUE_SECTORS, the default since 2026-11-23), which kept the full grid
    # before at every size and keeps it below the floor (_index_frame_cell).
    below_floor, cell_floor, cell_floor_switch = (
        _cell_below_sector_floor(
            problem,
            mpi_context,
            spin_polarized=spin_polarized,
            floor=periodic_floor,
            spin_floor=periodic_spin_floor,
            sector_forces=sector_forces and not _index_frame_cell(problem),
        )
        if periodic_symmetry and symmetry_mode != "on"
        else (None, periodic_floor, _PERIODIC_FLOOR_SWITCH)
    )
    if below_floor is not None:
        periodic_symmetry = False
    if mpi_hybrid is False:
        # (a hybrid of the ranks below the floor of its sectors, or under PARSEC_GAMMA_HYBRID_SECTORS=off: the one
        # sector of the identity, the full grid that one process takes there)
        periodic_symmetry = False
    if is_periodic and symmetry_mode == "on" and not periodic_symmetry:
        raise ValueError(
            "symmetry='on' needs a periodic cell whose point group the route of the sectors takes (a bulk "
            "orthorhombic cell at the Gamma point, unpolarized or collinear spin, native LDA or GGA, or TS, LDA+U, "
            "meta-GGA or libxc under PARSEC_GAMMA_TS_SECTORS and the like) and "
            "PARSEC_PERIODIC_SYMMETRY=auto"
            + ("" if periodic_refusal is None else f"; this calculation has {periodic_refusal}")
        )
    if (
        is_periodic
        and spin_polarized
        and mpi_context is not None
        and periodic_refusal is None
        and not periodic_symmetry
        # (the one sector of the identity of both channels that the runner's ranks prepare for spin moves without
        # a point group or after a spin guard fired, spin_identity_sector: it takes no group by design)
        and not spin_identity_sector
    ):
        # Without a detection the cell takes no group, and the identity sector of the ranks takes no spin channels:
        # refused before the reference is prepared.
        raise ValueError(
            spin_cell_group_refusal(
                "PARSEC_PERIODIC_SYMMETRY=off" if periodic_symmetry_mode == "off" else "it is prepared without symmetry"
            )
        )
    if periodic_refusal is not None and mpi_context is not None:
        raise ValueError(
            f"{periodic_refusal} of a periodic cell does not run as an MPI SCF, whose "
            "ranks share the basis of one sector of the identity (a bulk "
            "orthorhombic cell at the Gamma point, unpolarized or collinear spin, native LDA or GGA): "
            "run it with parsec_python/main.py"
            + (" --mpi" if "k-point" in periodic_refusal else "")
        )
    # Only the root rank of an MPI run enters the SCF loop.  The other ranks
    # answer eigensolver and density commands for the sectors they own.
    sector_worker = (
        mpi_context is not None and mpi_context.rank != mpi_context.root
    )
    from .Eigensolvers.symmetry import process_pool_min_bytes, process_sector_placement_setting

    # Read here whatever the run, so that a value they do not know is refused before the SCF; one process with
    # several GPUs places its sectors by them below (_process_sector_placement).
    process_sector_placement_setting()
    process_pool_min_bytes()
    # (and PARSEC_PERIODIC_SECTOR_DEVICES, the one GPU of the small sectors of a cell: _periodic_one_device_note)
    periodic_sector_devices_setting()
    from .Symmetry.sector_stencil import sector_stencil_route

    # The symmetry sectors take their packed stencils from the grid, or from
    # the cache, so the full-grid matrix is built only for a consumer that
    # reads it.  ``PARSEC_SECTOR_STENCIL=csr`` keeps the former route, which
    # builds the matrix first unless a cache may make it unnecessary.  The
    # switch is read here whatever the symmetry mode, so that a value it does
    # not know is refused in every run.
    stencil_route = sector_stencil_route()
    from .Symmetry.weighted_inputs import weighted_input_settings, weighted_route_settings

    # (and so are those of the weighted sectors of Fix_Number_Eigenvalues, libxc, occup.in and spin, and where
    # spin and occup.in take them: PARSEC_ORBITAL_GROUP_MPI, PARSEC_WEIGHTED_GUARD_SHORTCUT,
    # PARSEC_ORBITAL_GROUP_SPIN_FLOOR)
    weighted_input_settings()
    spin_floor = weighted_route_settings()[2]
    from .SCF.symmetry_fields import sector_input_group_average_setting

    # PARSEC_SECTOR_INPUT_GROUP_AVERAGE (the full-grid inputs of weighted sectors averaged over the full group):
    # read here for the same reason.
    input_group_average = sector_input_group_average_setting()
    from .Symmetry.field_orbits import sector_field_group_setting

    # PARSEC_SECTOR_FIELD_GROUP (the SCF fields of weighted sectors on the orbits of the full group).
    sector_field_group = sector_field_group_setting()
    from .Hartree.cupy_boundary import boundary_rows_setting

    # PARSEC_HARTREE_BOUNDARY_ROWS (read by the boundary builder on its thread).
    boundary_rows_setting()
    # PARSEC_IONIC_SUM_GROUP (the orbits of the GPU ionic sums on their thread).
    ionic_sum_group = ionic_sum_group_setting()
    # PARSEC_IONIC_MOVED_SUMS (the GPU ionic sums of moved structures), read for the same reason.
    moved_ionic_sum_setting()
    from .Eigensolvers.filter_graph import graph_reuse_requested, graph_route_name

    # Read here for the same reason: the first filter that records a graph
    # comes after the whole preparation, on every rank.
    shared_filter_graphs = graph_reuse_requested()
    from .Eigensolvers.distributed_state import ritz_pipeline, strided_exchange_requested

    # And so are the switches of the Ritz step of a sector basis that several
    # devices share: its first step comes after the preparation too.
    shared_ritz_pipeline = ritz_pipeline()
    shared_exchange_copies = "strided" if strided_exchange_requested() else "packed"
    if is_periodic or symmetry_mode == "off":
        # No symmetry is looked for: nobody waits for a reduction.  (A
        # periodic cell detects its point group in line below; its ionic
        # fields keep the sums over the whole cell.)
        preparation.symmetry_reduction.publish(None)
    elif _symmetry_overlap_requested() and not symmetry_constraints and not point_charges:
        # The reference stage starts it at the grid where it may.  (A group
        # with constraints, or with point charges, is detected in line below:
        # the thread knows the atoms alone.)
        preparation.symmetry_maps = _OverlappedSymmetryMaps(
            cache_directory=symmetry_cache_directory,
            reduction_known=preparation.symmetry_reduction,
            stages=stages,
        )
        preparation.symmetry_maps.point_group_wanted = (
            _orbital_group_refusal(problem, spin_polarized, mpi_context) is None
        )
        if spin_polarized and spin_floor > 0:
            # (decided again on the grid: the floor of the weighted spin sectors, as the detection in line decides)
            preparation.symmetry_maps.point_group_wanted_for = lambda grid: (
                _orbital_group_refusal(
                    problem, spin_polarized, mpi_context, basis=_spin_floor_basis(problem, grid)
                )
                is None
            )
        preparation.symmetry_maps.ionic_sum_group = ionic_sum_group
        if mpi_context is not None:
            preparation.symmetry_maps.ranks_divide = lambda reduction: _ranks_divide_sectors(reduction, mpi_context)
        preparation.symmetry_maps.fixed_atoms = symmetry_fixed_atoms
        preparation.join_on_exit(preparation.symmetry_maps.close)
    selection, reference, preparation_overlap = _resolve_and_prepare_reference(
        problem,
        backend,
        defer_native_laplacian=(
            symmetry_mode != "off"
            and (symmetry_cache_directory is not None or stencil_route != "csr")
        ),
        deferred_laplacian_cache_directory=symmetry_cache_directory,
        orbital_operators_only=sector_worker,
        # Every device of an MPI rank belongs to one of its sector groups.
        ionic_sector_rank=mpi_context is not None,
        **({} if ionic_fields is None else {"ionic_field_overlap": ionic_fields}),
        preparation=preparation,
    )
    # A few backend-wiring tests use a deliberately minimal reference object.
    # Preserve the historical CA default for such result-like objects while
    # real prepared systems always carry the parsed SCF functional.
    xc_functional = getattr(
        getattr(getattr(reference, "input", None), "scf", None),
        "xc_functional",
        "ca",
    )

    symmetry_reduction = None
    symmetry_geometry_cache_info = None
    symmetry_representation_cache_info = None
    detected_group_order = 1
    symmetry_detection = "disabled by option"
    undivided_note = None
    # (the NOTE of a spin or occup.in run that keeps the abelian group by PARSEC_ORBITAL_GROUP_MPI=off or the floor of
    # PARSEC_ORBITAL_GROUP_SPIN_FLOOR: _kept_abelian_note)
    orbital_group_note = None
    # (the NOTE of an LDA+U run whose units of PARSEC_ORBITAL_IRREPS keep the rows of their sectors)
    irrep_rows_note = None
    # the commuting signed permutations about the origin (the last group of the palette ladder below)
    origin_group = None
    # the frame of the grid indices of the detection of an oblique cell (PARSEC_OBLIQUE_SECTORS, Symmetry/oblique_frame.py)
    symmetry_frame = None
    symmetry_maps = preparation.symmetry_maps
    maps_beside = False
    # The atoms of the detection: the fixed atoms of a relaxation with Move_Flag as a species of their own
    # (PARSEC_MOVE_FLAG_SUBGROUP=auto), which gives the subgroup the relaxation keeps, and the points the operations
    # must keep besides them (the point charges of PARSEC_POINT_CHARGE_SYMMETRY=auto, the axis of a field:
    # symmetry_constraints), as pseudo-atoms of their own labels after the atoms.  The orbits of the atoms of a
    # weighted plan are those of the atoms alone.
    real_atoms = getattr(reference, "atoms", ())
    detection_atoms = _detection_atoms(real_atoms, symmetry_fixed_atoms)
    # (also the points the symmetry guard of a spin-polarized run in the sectors adds to the atoms: a field or
    # charges break the group of the atoms alone, which the densities then do not keep)
    detection_extra: tuple = ()
    if symmetry_mode != "off" and not is_periodic:
        from .SCF.symmetry_off_inputs import point_charge_constraints

        detection_extra = tuple(symmetry_constraints) + (
            point_charge_constraints(getattr(reference, "point_charges", ())) if point_charges else ()
        )
        if detection_extra:
            detection_atoms = tuple(detection_atoms) + detection_extra
    periodic_on_host = periodic_symmetry and selection.selected != "cupy"
    if periodic_on_host:
        # The sectors of a point group are a route of the CuPy backend: the
        # full grid of another backend keeps no orbits of the cell.
        if symmetry_mode == "on":
            raise ValueError(
                "symmetry='on' needs the CuPy backend for the point group of a periodic cell "
                f"(backend {selection.selected})"
            )
        periodic_symmetry = False
    if is_periodic and not periodic_symmetry:
        symmetry_detection = (
            "skipped: periodic cell ("
            + (
                "PARSEC_PERIODIC_SYMMETRY=off"
                if periodic_symmetry_mode == "off"
                else "disabled by option"
                if symmetry_mode == "off"
                else f"the sectors of its point group are a route of the CuPy backend, not of {selection.selected}"
                if periodic_on_host
                else (
                    f"{below_floor[0]} grid points x {below_floor[1]} states are below the floor of the sectors, "
                    f"{cell_floor} entries ({cell_floor_switch})"
                )
                if below_floor is not None
                else (
                    "a hybrid of the MPI ranks whose grid points x states are below the 2^24 of its sectors, or "
                    "PARSEC_GAMMA_HYBRID_SECTORS=off"
                )
                if mpi_hybrid is False and periodic_refusal is None
                else f"the route of the sectors does not take {periodic_refusal}"
            )
            + "; the full grid is used)"
        )
    elif symmetry_mode != "off":
        from .Symmetry import load_or_detect_reflection_reduction

        # The thread that works on the maps since the grid exists, where the
        # reference stage started one on this grid and these atoms.
        maps_beside = symmetry_maps is not None and symmetry_maps.serves(
            reference.grid, detection_atoms
        )
        try:
            # A cell whose grid axes are not the Cartesian axes (PARSEC_OBLIQUE_SECTORS=auto): its detection in the
            # frame of the grid indices (Symmetry/oblique_frame.py: the atoms at S^-1 r, a grid of the same rows with
            # the box of its counts as cell, the admissible operations); every other grid as it is.
            symmetry_frame = (
                _symmetry_index_frame(reference.grid, problem, mpi_context) if is_periodic else None
            )
            detection_grid = reference.grid if symmetry_frame is None else symmetry_frame.frame_grid
            frame_atoms = detection_atoms if symmetry_frame is None else symmetry_frame.atoms(detection_atoms)
            frame_tolerance = atom_tolerance if symmetry_frame is None else symmetry_frame.tolerance(atom_tolerance)
            if maps_beside:
                candidate, symmetry_geometry_cache_info = symmetry_maps.reduction()
            else:
                candidate, symmetry_geometry_cache_info = stages.timed(
                    "symmetry_reduction", load_or_detect_reflection_reduction
                )(
                    detection_grid,
                    frame_atoms,
                    # (no symmetry cache in the frame of an oblique cell: its key holds the indices, not the lattice
                    # whose operations are admissible)
                    cache_directory=symmetry_cache_directory if symmetry_frame is None else None,
                    atom_tolerance=frame_tolerance,
                )
        except (ValueError, RuntimeError) as error:
            if symmetry_mode == "on":
                raise ValueError(
                    "symmetry='on' requested, but exact symmetry detection "
                    f"failed: {error}"
                ) from error
            symmetry_detection = (
                f"automatic detection fell back to full grid: "
                f"{type(error).__name__}: {error}"
            )
        else:
            detected_group_order = candidate.group_order
            refusal = _orbital_group_refusal(
                problem,
                spin_polarized,
                mpi_context,
                # (counted only where a floor is given: none under auto, SPIN_FLOOR_ENTRIES = 0)
                basis=_spin_floor_basis(problem, reference.grid) if spin_polarized and spin_floor > 0 else None,
            )
            if candidate.group_order > 1 or is_periodic:
                # A spin or occup.in run that keeps the abelian group by the rules of 2026-11-21 says so at every
                # output level (PARSEC_ORBITAL_GROUP_MPI=off, PARSEC_ORBITAL_GROUP_SPIN_FLOOR).
                orbital_group_note = _kept_abelian_note(refusal)
            # A periodic cell looks for its group even where its abelian group about the origin is the identity
            # alone: the half-cell translations of a cell that repeats a smaller one, the operations about a
            # centre off the origin (PARSEC_PERIODIC_TRIVIAL_ORIGIN=off: only where that group is larger).
            search_cell = is_periodic and trivial_origin_mode == "auto"
            # The group about the origin, kept where the MPI ranks cannot divide the sectors of a larger one
            # (_ranks_divide_sectors), which is then named in the report.
            origin_group, undivided = candidate, None
            if (candidate.group_order > 1 or search_cell) and refusal is None:
                # The one-dimensional sectors of a larger subgroup of the full point group, with weights,
                # where they solve less (Symmetry/orbital_groups.py; PARSEC_ORBITAL_GROUP=abelian keeps
                # the group of before): detected by the thread of the maps where it was asked to
                # (PARSEC_OVERLAP_SYMMETRY_SETUP), in line otherwise.
                found = symmetry_maps.point_group() if maps_beside else None
                if found is not None:
                    candidate = found
                else:
                    from .Symmetry import detect_point_group_reduction

                    candidate = stages.timed("orbital_group", detect_point_group_reduction)(
                        detection_grid,
                        frame_atoms,
                        candidate,
                        atom_tolerance=frame_tolerance,
                        # the fixed cost of a sector of a cell in one process (PARSEC_SECTOR_OVERHEAD)
                        sector_cost=_periodic_sector_cost(problem, reference, mpi_context) if is_periodic else 0.0,
                    )
                    if detection_atoms is not real_atoms:
                        from .SCF.symmetry_off_inputs import orbits_of_real_atoms

                        orbits_of_real_atoms(candidate, len(real_atoms))
                if candidate is not origin_group and not _ranks_divide_sectors(candidate, mpi_context):
                    candidate, undivided = origin_group, candidate
            if symmetry_frame is not None and symmetry_frame.general:
                from .Symmetry.axis_reflection import holds_general_operations

                if not holds_general_operations(candidate):
                    # PARSEC_OBLIQUE_FULL_GROUP=auto, where the sectors of the signed permutations solve as little (or
                    # no weighted plan is taken): the frame of the signed permutations alone, as before
                    symmetry_frame = symmetry_frame.signed()
                    detection_grid = symmetry_frame.frame_grid
            from .Symmetry import PointGroupReduction

            if is_periodic and (candidate.group_order > 1 or search_cell) and not isinstance(
                candidate, PointGroupReduction
            ):
                # The orbitals of a cell that keep an abelian group take its commuting involutions with their
                # translations (PARSEC_PERIODIC_ABELIAN_TRANSLATIONS=off: those about the origin, as before).
                from .Symmetry.axis_reflection import periodic_abelian_reduction

                abelian_group = stages.timed("orbital_group", periodic_abelian_reduction)(
                    detection_grid, frame_atoms, candidate, atom_tolerance=frame_tolerance
                )
                if abelian_group is candidate or _ranks_divide_sectors(abelian_group, mpi_context):
                    candidate = abelian_group
                elif undivided is None:
                    undivided = abelian_group
            if undivided is not None:
                undivided_note = (
                    f"the {_sector_count(undivided)} sectors of the larger group found are fewer than the "
                    f"{mpi_context.size} MPI ranks, which take one or more each (ranks share a sector with one GPU "
                    f"per rank): the orbitals keep the group about the origin"
                )
            if candidate.group_order > 1:
                dropped_note = None
                if _sectors_keep_orbits(reference, ldau_orbits, candidate, frame_atoms, grid=detection_grid):
                    # Abelian sectors take the projectors of one atom per orbit, as those of a point group do
                    # (PARSEC_SECTOR_PROJECTOR_ORBITS; the device LDA+U columns follow the KB columns of their
                    # atom, and their coefficients those of the representatives, PARSEC_LDAU_PROJECTOR_ORBITS).
                    from .Symmetry.axis_reflection import attach_abelian_atom_orbits

                    attach_abelian_atom_orbits(candidate, frame_atoms, detection_grid)
                    if len(detection_atoms) > len(real_atoms):
                        # the pseudo-atoms of the detection (point charges, a field axis, the constraints of
                        # PARSEC_MOVING_SECTORS) after the atoms have orbits of their own labels: the orbits of the
                        # atoms alone, as for a point group above (the projectors, their gradients and the check of
                        # moved atoms read one orbit per atom)
                        from .SCF.symmetry_off_inputs import orbits_of_real_atoms

                        orbits_of_real_atoms(candidate, len(real_atoms))
                else:
                    # LDA+U under PARSEC_LDAU_PROJECTOR_ORBITS=off, or atoms off the group by more than 1e-6 bohr
                    # (a loose PARSEC_SYMMETRY_ATOM_TOLERANCE, with or without LDA+U): the projectors of every atom
                    # (with LDA+U the fifteen columns of every LDA+U atom that Hamiltonian/hubbard_device.py writes),
                    # also where the group of a cell knows its atom orbits
                    if getattr(candidate, "atom_representatives", None) is not None:
                        if getattr(reference, "hubbard", None) is None:
                            dropped_note = _SECTOR_LOOSE_ORBITS_NOTE
                        elif ldau_orbits == "auto":
                            dropped_note = _LDAU_LOOSE_ORBITS_NOTE
                    candidate = _without_atom_orbits(candidate)
                symmetry_reduction = candidate
                symmetry_detection = _symmetry_detection_text(candidate)
                if dropped_note is not None:
                    symmetry_detection += "; " + dropped_note
            else:
                symmetry_detection = (
                    "identity only; no nontrivial supported symmetry detected"
                )
                if symmetry_mode == "on":
                    raise ValueError(
                        "symmetry='on' requested, but only the identity "
                        "operation preserves the labeled atoms and active grid"
                    )
            from .Symmetry.axis_reflection import LOOSE_ATOM_TOLERANCE, lost_operations_note

            # Operations the atoms keep only to more than the atom tolerance (PARSEC_SYMMETRY_ATOM_TOLERANCE):
            # said in the report, since the positions are not symmetrized.
            try:
                lost_note = lost_operations_note(
                    detection_grid,
                    frame_atoms,
                    candidate,
                    atom_tolerance=frame_tolerance,
                    **(
                        {}
                        if symmetry_frame is None
                        else {
                            # (the frame's units in the search, bohr in the line)
                            "loose_tolerance": symmetry_frame.tolerance(LOOSE_ATOM_TOLERANCE),
                            "frame": symmetry_frame,
                            "reported_tolerances": (atom_tolerance, LOOSE_ATOM_TOLERANCE),
                        }
                    ),
                )
            except (ValueError, RuntimeError, AttributeError):
                lost_note = None
            if lost_note is not None:
                symmetry_detection += "; " + lost_note
            if (
                is_periodic
                and candidate.group_order > 1
                and (
                    getattr(candidate, "full_translations", None) is not None
                    or getattr(candidate, "translations", None) is not None
                )
            ):
                from .Symmetry.axis_reflection import defect_centre_note

                # A group whose operations about a defect came from its rarest atom environment
                # (PARSEC_PERIODIC_DEFECT_CENTRE): named with the site, since it is not the origin of the cell.
                try:
                    defect_note = defect_centre_note(
                        detection_grid,
                        (
                            getattr(reference, "atoms", ())
                            if symmetry_frame is None
                            else symmetry_frame.atoms(getattr(reference, "atoms", ()))
                        ),
                        atom_tolerance=frame_tolerance,
                        frame=symmetry_frame,
                    )
                except (ValueError, RuntimeError, AttributeError):
                    defect_note = None
                if defect_note is not None:
                    symmetry_detection += "; " + defect_note
            if undivided_note is not None:
                symmetry_detection += "; " + undivided_note
            if symmetry_fixed_atoms is not None:
                fixed_count = sum(bool(flag) for flag in symmetry_fixed_atoms)
                symmetry_detection += (
                    f"; the {fixed_count} atom{'s' if fixed_count != 1 else ''} that Move_Flag fixes count as a "
                    "species of their own: the operations map them onto fixed atoms (PARSEC_MOVE_FLAG_SUBGROUP=auto)"
                )
    # The molecular dynamics and manual movements of a cell under PARSEC_MOVING_SECTORS=auto take sectors of
    # minimum_sector_order operations or more: dropped here, before they are built (SCF/moving_sectors.py).
    if is_periodic and symmetry_mode != "on" and _below_minimum_sector_order(symmetry_reduction, minimum_sector_order):
        symmetry_detection = _minimum_sector_order_text(symmetry_reduction, minimum_sector_order, symmetry_detection)
        symmetry_reduction = None
    # A wire or slab whose solved sectors are too few or too small (PARSEC_PERIODIC_SECTOR_FLOOR=auto,
    # _wire_slab_sector_shortfall): dropped here, before they are built; it runs on the full grid as below the floor.
    if is_periodic and symmetry_mode != "on" and symmetry_reduction is not None:
        shortfall = _wire_slab_sector_shortfall(
            problem, reference.grid, symmetry_reduction, mpi_context, sector_forces=sector_forces
        )
        if shortfall is not None:
            symmetry_detection = f"skipped: periodic cell ({shortfall}; the full grid is used)"
            symmetry_reduction = None
    # A cell whose sector stencils would need more coefficients than their palette holds (_sector_palette_check):
    # a smaller group whose stencils fit, or the full grid, decided before the identity sector of an MPI run is
    # chosen, from the same inputs on every rank (PARSEC_SECTOR_PALETTE_LADDER).
    palette_checked = None
    palette_note = None
    # (read in every run, so that a value they do not know is refused)
    palette_ladder = sector_palette_ladder_setting()
    from .backends.cupy_stencil_major import sector_palette_codes_setting

    palette_codes = sector_palette_codes_setting()
    if (
        is_periodic
        and symmetry_reduction is not None
        and selection.selected == "cupy"
        and not (spin_polarized and spin_sector_refusal is not None)
    ):
        from .Symmetry import load_or_build_reflection_decomposition
        from .Symmetry.sector_stencil import sector_palette_bounds, sector_palette_sizes

        def palette_decomposition(orbital):
            # what the stage of the orbitals below builds, or takes from the thread of the maps
            built = symmetry_maps.decomposition(orbital) if maps_beside else None
            if built is None:
                built = stages.timed("symmetry_decomposition", load_or_build_reflection_decomposition)(
                    # (the row maps of an oblique cell's index operations: the grid of its frame, the same rows)
                    reference.grid if symmetry_frame is None else symmetry_frame.frame_grid,
                    orbital,
                    reduction_key=getattr(symmetry_geometry_cache_info, "key", None),
                    cache_directory=symmetry_cache_directory,
                )
            return built

        def palette_check(reduction):
            return _sector_palette_check(
                reference.grid,
                reduction,
                palette_decomposition,
                stages.timed("sector_palette_check", sector_palette_sizes),
                stages.timed("sector_palette_bound", sector_palette_bounds),
            )

        def lower_groups():
            # The groups below the one detected whose stencils may fit: the commuting operations with their
            # translations, below the weighted sectors of a point group; the commuting signed permutations about
            # the origin, which keep the palette small (D2 of a cubic crystal: 40 values at order 12, 123 at 20).
            from .Symmetry import PointGroupReduction

            if isinstance(symmetry_reduction, PointGroupReduction) and origin_group is not None:
                from .Symmetry.axis_reflection import periodic_abelian_reduction

                try:
                    abelian = stages.timed("orbital_group", periodic_abelian_reduction)(
                        detection_grid, frame_atoms, origin_group, atom_tolerance=frame_tolerance
                    )
                except (ValueError, RuntimeError):
                    abelian = None
                if abelian is not None and abelian is not origin_group and abelian.group_order > 1:
                    yield abelian
            if origin_group is not None and origin_group is not symmetry_reduction and origin_group.group_order > 1:
                yield origin_group

        palette_checked, palette_refusal = palette_check(symmetry_reduction)
        if palette_checked is not None and palette_checked[2] > 256:
            # two-byte codes keep the group (PARSEC_SECTOR_PALETTE_CODES=auto; off steps down the ladder below)
            symmetry_detection += (
                f"; the sector stencils of {_palette_group_label(palette_checked[0])} need {palette_checked[2]} "
                f"distinct coefficients and code them in two bytes (PARSEC_SECTOR_PALETTE_CODES={palette_codes})"
            )
        if palette_refusal is not None:
            refusals = [palette_refusal]
            chosen = None
            if palette_ladder == "auto":
                for lower in lower_groups():
                    if not _ranks_divide_sectors(lower, mpi_context):
                        continue
                    checked, refusal = palette_check(lower)
                    if refusal is None:
                        palette_checked, chosen = checked, lower
                        break
                    refusals.append(refusal)
            if chosen is not None:
                if _sectors_keep_orbits(reference, ldau_orbits, chosen, frame_atoms, grid=detection_grid):
                    from .Symmetry.axis_reflection import attach_abelian_atom_orbits

                    attach_abelian_atom_orbits(chosen, frame_atoms, detection_grid)
                else:
                    chosen = _without_atom_orbits(chosen)
                palette_note = (
                    "; ".join(refusals)
                    + f": the orbitals take {_palette_group_label(chosen)}, whose stencils fit "
                    "(PARSEC_SECTOR_PALETTE_LADDER=auto)"
                )
                symmetry_detection = (
                    f"{_symmetry_detection_text(chosen)}; {palette_note}; found first: {symmetry_detection}"
                )
                symmetry_reduction = chosen
            else:
                if symmetry_mode == "on":
                    raise ValueError(f"symmetry='on' requested, but {'; '.join(refusals)}")
                palette_note = (
                    "; ".join(refusals)
                    + ": the orbitals take the full grid"
                    + (", one sector of the identity" if mpi_context is not None else "")
                    + (
                        " (PARSEC_SECTOR_PALETTE_LADDER=off)"
                        if palette_ladder == "off"
                        else " (PARSEC_SECTOR_PALETTE_LADDER)"
                    )
                )
                symmetry_detection = (
                    f"automatic detection fell back to full grid: {'; '.join(refusals)}; found: {symmetry_detection}"
                )
                symmetry_reduction = None
    # (a smaller group of the palette ladder, under the minimum_sector_order of PARSEC_MOVING_SECTORS=auto)
    if is_periodic and symmetry_mode != "on" and _below_minimum_sector_order(symmetry_reduction, minimum_sector_order):
        symmetry_detection = _minimum_sector_order_text(symmetry_reduction, minimum_sector_order, symmetry_detection)
        symmetry_reduction = None
        palette_note = None
    # (a smaller group of the palette ladder of a wire or slab, under the rule of _wire_slab_sector_shortfall)
    if is_periodic and symmetry_mode != "on" and symmetry_reduction is not None:
        shortfall = _wire_slab_sector_shortfall(
            problem, reference.grid, symmetry_reduction, mpi_context, sector_forces=sector_forces
        )
        if shortfall is not None:
            symmetry_detection = f"skipped: periodic cell ({shortfall}; the full grid is used)"
            symmetry_reduction = None
            palette_note = None
    if symmetry_frame is not None and symmetry_reduction is not None:
        # The operations of this reduction are the signed permutations W of the grid indices of an oblique cell
        # (Symmetry/oblique_frame.py; the Cartesian ones are S W S^-1): kept on it for the readers that turn vectors
        # (the forces refuse such sectors, Forces/compute.py).
        object.__setattr__(symmetry_reduction, "index_frame", symmetry_frame)
        if symmetry_frame.general:
            symmetry_detection += (
                f"; detected in the frame of the grid indices (PARSEC_OBLIQUE_SECTORS=auto: {symmetry_frame.order} "
                f"operations of the grid indices are isometries of the cell that keep its Laplacian, "
                f"{symmetry_frame.order - len(symmetry_frame.general)} signed permutations of the grid axes and "
                f"{len(symmetry_frame.general)} other integer maps, PARSEC_OBLIQUE_FULL_GROUP=auto for a functional "
                "without a density gradient)"
            )
        else:
            symmetry_detection += (
                f"; detected in the frame of the grid indices (PARSEC_OBLIQUE_SECTORS=auto: {symmetry_frame.order} "
                "signed permutations of the grid axes are isometries of the cell that keep its finite-difference "
                "operators)"
            )
    # Without a point group the orbitals may still take the route of the
    # sectors, as one sector of the identity, so that the devices of the
    # process share its basis (PARSEC_CUPY_SINGLE_SECTOR).
    single_sector_mode = single_sector_setting()
    # (read for every input: a value it does not know stops the run here, whatever sectors it takes)
    from .Eigensolvers.symmetry import identity_sector_states_setting

    identity_sector_states_setting()
    # (the full grid of both spin channels of a fallback, shared: spin_identity_sector)
    single_sector = (spin_identity_sector and symmetry_reduction is None and selection.selected == "cupy") or (
        (
            single_sector_mode == "on"
            or (single_sector_mode == "auto" and mpi_context is not None)
        )
        and symmetry_reduction is None
        # A periodic cell where the route of the sectors takes it (a bulk
        # orthorhombic cell at Gamma on the grid of one spacing).
        and (
            not is_periodic
            or (
                periodic_refusal is None
                and not getattr(reference.grid, "is_general_periodic", False)
            )
        )
        and not spin_polarized
        and selection.selected == "cupy"
    )
    if (
        is_periodic
        and spin_polarized
        and mpi_context is not None
        and symmetry_reduction is None
        and not getattr(reference.grid, "is_general_periodic", False)
        # (not the one sector of the identity of both channels that the runner's ranks prepare together after a
        # spin guard fired, PARSEC_SPIN_FALLBACK_SHARED: that preparation takes no group by design)
        and not spin_identity_sector
    ):
        # The detection (or the palette ladder) left it no group: refused here, before the Hartree, the sectors and
        # the CUDA work of the preparation, with the reason of a spin cell (single_sector takes no spin channels).
        raise ValueError(
            spin_cell_group_refusal(
                f"it keeps no group of more than one operation here (symmetry_detection: {symmetry_detection})"
            )
        )
    if is_periodic and mpi_context is not None and not single_sector and symmetry_reduction is None:
        raise ValueError(
            "the MPI SCF of a periodic cell solves it in the sectors of its point group or as one sector "
            "of the identity, which this run "
            + (
                "cannot take on Fortran's general periodic grid (Use_Fortran_Routines with sides "
                "of different grid steps)"
                if getattr(reference.grid, "is_general_periodic", False)
                else f"does not take here (PARSEC_CUPY_SINGLE_SECTOR={single_sector_mode}, "
                f"backend {selection.selected})"
            )
            + ": run it with parsec_python/main.py"
        )
    if single_sector and mpi_context is not None and not spin_identity_sector:
        # One sector shared by ranks that hold the GPUs of a node each would lie on the first GPU of every rank
        # (experimental/mpi_scf.single_sector_layout_refusal, PARSEC_SINGLE_SECTOR_NODE_LAYOUT): refused here, right
        # after the detection that found no group, before the sectors are built.  Not the shared full grid of a
        # spin guard fallback (spin_identity_sector, PARSEC_SPIN_FALLBACK_SHARED): its plan counts one GPU per rank
        # already, it replaces the full grid of the root's one GPU of before, and refusing it would discard the
        # sector SCF that ran first (MPISectorContext.spin_fallback_layout exempts it in configure as well).
        layout_refusal = getattr(mpi_context, "single_sector_layout_refusal", None)
        if callable(layout_refusal):
            from .Eigensolvers.symmetry import selected_device_ids
            from .backends.cupy import require_cupy

            cp, _ = require_cupy()
            refusal = layout_refusal(len(selected_device_ids(int(cp.cuda.Device().id))))
            if refusal is not None:
                raise ValueError(refusal)
    if single_sector:
        from .Symmetry import AxisReflectionReduction

        symmetry_reduction = AxisReflectionReduction.identity(reference.grid)
        symmetry_detection = (
            f"{symmetry_detection}; one sector of the identity "
            + (
                "for both spin channels, its basis shared by "
                + ("the MPI ranks" if mpi_context is not None else "the GPUs of the process")
                + " (the full grid of both spin channels where one device would not hold it, "
                "PARSEC_SPIN_FALLBACK_SHARED)"
                if spin_identity_sector
                else f"(PARSEC_CUPY_SINGLE_SECTOR={single_sector_mode})"
            )
        )
    if is_periodic and symmetry_reduction is not None:
        from .SCF.gamma_features import taken_features

        # TS, LDA+U, a meta-GGA or libxc that their switches brought into the sectors (SCF/gamma_features.py)
        taken = taken_features(problem)
        if taken:
            from .SCF.gamma_features import gamma_feature_setting

            symmetry_detection = f"{symmetry_detection}; with " + ", ".join(
                f"{name}={gamma_feature_setting(name)}" for name in taken
            )
    # The reduction of this preparation is decided: a thread that waits for
    # it goes on (the thread of the maps has published it where it ran).  The
    # ionic sums take the orbits of the full point group of weighted sectors
    # (PARSEC_IONIC_SUM_GROUP=full).
    preparation.symmetry_reduction.publish(
        # (the orbits of the full group, then those of H and of the abelian group detected first)
        _ionic_sum_candidates(
            symmetry_reduction,
            stages,
            first=origin_group if hasattr(symmetry_reduction, "coset_images") else None,
        )
        if ionic_sum_group == "full" and ionic_fields is not None
        else symmetry_reduction
    )

    # Establish whether the orbital problem can use exact representation
    # sectors before constructing the GPU backend.  This ordering matters:
    # a sector calculation never applies the full-grid GPU Hamiltonian, so
    # uploading that duplicate allocation would be pure setup and memory cost.
    orbital_decomposition = None
    # the filter plan of the input where the first shares of weighted sectors asked for it (asked once: below)
    share_filter_plan = _NOT_PLANNED
    # A/B control and conservative architecture override: with the
    # stabilizer policy off Hartree retains the detected scalar reduction of
    # a group that fixes grid points, while orbitals use the full grid.
    orbital_reduction = _orbital_reduction(
        symmetry_reduction, _stabilizer_symmetry_disabled()
    )
    orbital_symmetry = "full grid"
    if symmetry_mode == "off" and not single_sector:
        orbital_symmetry = "full grid (symmetry disabled)"
    elif orbital_reduction is None or (
        orbital_reduction.group_order <= 1 and not single_sector
    ):
        orbital_symmetry = "full grid (no nontrivial supported symmetry)"
    elif selection.selected != "cupy":
        orbital_symmetry = (
            "full grid (representation decomposition is currently a CuPy path)"
        )
    elif spin_polarized and spin_sector_refusal is not None:
        orbital_symmetry = f"full grid (spin-polarized: {spin_sector_refusal})"
    else:
        from .Symmetry import load_or_build_reflection_decomposition

        try:
            # The decomposition the thread of the maps built of this very
            # reduction, where it built one (or the check of the palette
            # above); in line otherwise.
            built = (
                palette_checked[1]
                if palette_checked is not None and palette_checked[0] is orbital_reduction
                else symmetry_maps.decomposition(orbital_reduction)
                if maps_beside
                else None
            )
            if built is None:
                built = stages.timed(
                    "symmetry_decomposition", load_or_build_reflection_decomposition
                )(
                    # (the row maps of an oblique cell's index operations: the grid of its frame, the same rows)
                    reference.grid if symmetry_frame is None or single_sector else symmetry_frame.frame_grid,
                    orbital_reduction,
                    reduction_key=getattr(symmetry_geometry_cache_info, "key", None),
                    cache_directory=None if single_sector else symmetry_cache_directory,
                )
            orbital_decomposition, symmetry_representation_cache_info = built
        except (ValueError, RuntimeError) as error:
            if symmetry_mode == "on":
                raise ValueError(
                    "symmetry='on' requested, but the GPU orbital "
                    f"representation decomposition is unusable: {error}"
                ) from error
            orbital_symmetry = (
                "full grid (automatic representation fallback: "
                f"{type(error).__name__}: {error})"
            )
        else:
            # LDA+U forms its occupation matrices, and a meta-GGA its kinetic density (meta_gga_device, the weighted
            # sectors of PARSEC_ORBITAL_GROUP_META_GGA), from the sector orbitals on the rows of their sectors: the
            # units of PARSEC_ORBITAL_IRREPS keep those rows under the partner default of PARSEC_ORBITAL_IRREP_ROWS.
            # Decided here, from the input, before the placement reads the solved sizes, alike on every MPI rank.
            rows_kept_for = None
            if getattr(orbital_decomposition, "partner_units", ()):
                if any(
                    getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values()
                ):
                    rows_kept_for = ("LDA+U", "LDA+U")
                elif getattr(getattr(problem, "scf", None), "is_meta_gga", False):
                    rows_kept_for = ("a meta-GGA", "A meta-GGA")
            if rows_kept_for is not None:
                reason, subject = rows_kept_for
                partner_units = tuple(orbital_decomposition.partner_units)
                orbital_decomposition.keep_sector_rows(reason)
                clause = _partner_rows_note(orbital_decomposition.reduction)
                if clause and isinstance(symmetry_detection, str):
                    symmetry_detection = symmetry_detection.replace(
                        clause, f", on the rows of their sectors for {reason} (PARSEC_ORBITAL_IRREP_ROWS)"
                    )
                irrep_rows_note = (
                    f"{subject} keeps the units {', '.join(str(index + 1) for index in partner_units)} of "
                    "PARSEC_ORBITAL_IRREPS on the rows of their sectors (PARSEC_ORBITAL_IRREP_ROWS=partner is not "
                    f"taken with {reason})"
                )
            sector_sizes = orbital_decomposition.sector_sizes
            # The first counts of weighted sectors from their shares of the valence orbitals of the atoms
            # (PARSEC_WEIGHTED_FIRST_SHARE: auto, the default since 2026-11-21, takes them in one process from a
            # full-grid basis of sector_shares.MINIMUM_ENTRIES on and keeps the even share with its margin on the ranks
            # of an MPI run and below it; even keeps it everywhere, as before).
            from .Symmetry.axis_reflection import symmetry_atom_tolerance
            from .Symmetry.sector_shares import attach_valence_orbital_counts, weighted_first_share_setting

            share_plan = share_filter_plan = _filter_plan(reference)
            # (a grid without a size, a stand-in of the tests, counts as not known: the even share)
            grid_points = getattr(reference.grid, "size", None)
            # (an oblique cell: its atoms and grid in the frame of its indices, whose signed permutations W have the
            # characters of their Cartesian R = S W S^-1, a matrix of the same trace and determinant)
            valence_counts = attach_valence_orbital_counts(
                orbital_decomposition,
                (
                    tuple(getattr(reference, "atoms", ()) or ())
                    if symmetry_frame is None
                    else symmetry_frame.atoms(tuple(getattr(reference, "atoms", ()) or ()))
                ),
                getattr(reference, "pseudopotentials", None),
                reference.grid if symmetry_frame is None else symmetry_frame.frame_grid,
                tolerance=(
                    symmetry_atom_tolerance()
                    if symmetry_frame is None
                    else symmetry_frame.tolerance(symmetry_atom_tolerance())
                ),
                mpi=mpi_context is not None,
                entries=(
                    None
                    if share_plan is None or grid_points is None
                    else int(grid_points) * (int(share_plan.states) + int(share_plan.buffer))
                ),
            )
            orbital_symmetry = (
                "CuPy real one-dimensional representations with exact "
                "orbit-stabilizer character selection"
                if len(set(sector_sizes)) > 1
                else "CuPy real one-dimensional reflection representations"
            )
            if valence_counts is not None:
                orbital_symmetry += (
                    "; first state counts from the shares of the valence orbitals of the atoms per sector, "
                    f"{', '.join(str(int(count)) for count in valence_counts)} "
                    f"(PARSEC_WEIGHTED_FIRST_SHARE={weighted_first_share_setting()})"
                )
    if symmetry_maps is not None:
        # The thread of the maps has ended here at the latest, whether its
        # decomposition was asked for or not: before the sector operators
        # are built and long before the first graph capture.
        symmetry_maps.close()
    if is_periodic and symmetry_reduction is not None and orbital_decomposition is None:
        if mpi_context is not None:
            # The ranks of a periodic cell divide the sectors of its group:
            # without them there is no route for this run.
            raise ValueError(
                f"the point group of this periodic cell was found but its orbitals stay on the {orbital_symmetry}, "
                "which the MPI SCF of a periodic cell does not solve: run it with PARSEC_PERIODIC_SYMMETRY=off"
            )
        # The full grid of one device, as before the point group of a cell:
        # no field of this calculation keeps the orbits of the group.
        symmetry_reduction = None
    # Spin_Symmetry_Sectors: both channels in the sectors of this
    # decomposition, with the symmetry guard of SCF/spin_sectors.py.
    spin_sectors = spin_polarized and orbital_decomposition is not None
    if spin_identity_sector and not spin_sectors:
        # (the shared layout was chosen because one device would not hold that full grid)
        raise ValueError(
            "the full grid of both spin channels as one sector of the identity shared by several GPUs "
            f"(PARSEC_SPIN_FALLBACK_SHARED) cannot be prepared: its orbitals stay on the {orbital_symmetry}"
        )
    if spin_polarized and mpi_context is not None and not spin_sectors:
        raise NotImplementedError(
            f"a spin-polarized calculation on the {orbital_symmetry} cannot run as an MPI SCF: "
            "the ranks divide the symmetry sectors of both channels. Run it in one process "
            "(--serial-control or main.py)"
        )

    # Decide the Hartree reduction before constructing the execution backend.
    # This lets the independent boundary-geometry setup overlap the much more
    # expensive GPU orbital-operator construction below.
    hartree_linear_backend, hartree_linear_reason = _hartree_linear_backend(
        selection, periodic=is_periodic
    )
    # Whether the variable named the backend: only a choice of ``auto`` may
    # fall back to the host CG where the GPU CG cannot be built.
    hartree_linear_named = hartree_linear_reason.startswith(
        "PARSEC_HARTREE_LINEAR_BACKEND="
    )
    hartree_reduction = symmetry_reduction
    legacy_hartree_setting = os.environ.get(
        "PARSEC_HARTREE_SYMMETRY", "auto"
    ).strip().lower()
    legacy_hartree_disabled = legacy_hartree_setting in {
        "0", "false", "no", "off"
    }
    legacy_hartree_forced = legacy_hartree_setting in {
        "1", "true", "yes", "on"
    }
    if selection.hartree_backend != "native":
        hartree_reduction = None
    elif (symmetry_mode == "off" and not single_sector) or (
        symmetry_mode == "auto" and legacy_hartree_disabled
    ):
        hartree_reduction = None
    elif legacy_hartree_forced and hartree_reduction is None:
        raise ValueError(
            "PARSEC_HARTREE_SYMMETRY requests reduction, but no nontrivial "
            "supported symmetry was detected"
        )
    hartree_full_grid_for_spin = False
    if spin_polarized and not spin_sectors and hartree_reduction is not None:
        # The orbitals of the two channels are solved on the full grid, where
        # a partly filled degenerate level may take a density that the point
        # group does not leave invariant (the oxygen atom).  The wedge solver
        # would average that density over the symmetry orbits and return the
        # Hartree potential of another density.  (In the sectors every
        # channel density is invariant: Spin_Symmetry_Sectors.)
        if legacy_hartree_forced:
            raise ValueError(
                "PARSEC_HARTREE_SYMMETRY requests the symmetry-reduced Hartree "
                "solver, which a spin-polarized calculation cannot use: its "
                "density need not keep the symmetry of the structure"
            )
        hartree_reduction = None
        hartree_full_grid_for_spin = True
    # Weighted sectors of a subgroup H: the density and every potential are invariant under the full group G, and
    # keep one value per orbit of G (PARSEC_SECTOR_FIELD_GROUP=full, Symmetry/field_orbits.py), [G:H] times fewer
    # than per orbit of H; the Hartree solve of the sectors' wedge takes the same orbits.  The same decision on
    # every MPI rank (the reduction and the switch are the same there).
    field_reduction = None
    if (
        sector_field_group == "full"
        and orbital_decomposition is not None
        and not spin_polarized
        and hasattr(orbital_reduction, "coset_images")
    ):
        from .Symmetry.field_orbits import field_orbit_reduction

        try:
            field_reduction = stages.timed("field_orbits", field_orbit_reduction)(orbital_reduction)
        except (ValueError, RuntimeError):
            # Orbits that do not form the full group: the fields keep those of the sectors.
            field_reduction = None
        if field_reduction is not None and hartree_reduction is orbital_reduction:
            hartree_reduction = field_reduction
    # What of a cluster with a point group runs on the full grid only because the
    # native extension is not used (``symmetry_full_grid_note`` of the details,
    # a NOTE line of parsec.out and of the command line's stderr).
    full_grid_notes = _cluster_full_grid_notes(
        selection,
        periodic=is_periodic,
        group_found=symmetry_reduction is not None and int(getattr(symmetry_reduction, "group_order", 1)) > 1,
        orbitals_in_sectors=orbital_decomposition is not None,
    )
    if palette_note is not None:
        # the group the palette of the sector stencils left the orbitals (_sector_palette_check), at every output level
        full_grid_notes.append(palette_note)
    if orbital_group_note is not None and symmetry_reduction is not None:
        # a spin or occup.in run that keeps the abelian group (_kept_abelian_note), at every output level
        full_grid_notes.append(orbital_group_note)
    if irrep_rows_note is not None and orbital_decomposition is not None:
        # LDA+U with units of PARSEC_ORBITAL_IRREPS on the rows of their sectors, at every output level
        full_grid_notes.append(irrep_rows_note)
    # A spin run in the sectors: where the full grid of both channels would go if its guard fired, and a NOTE where it
    # would not fit one device (PARSEC_SPIN_FALLBACK_SHARED; the shared full grid of a fallback is that full grid).
    spin_fallback = None
    if spin_sectors and not spin_identity_sector and not sector_worker:
        # (the root alone goes back to the full grid; the other ranks of the runner follow its reprepare command)
        spin_fallback = spin_fallback_plan(reference, mpi_context=mpi_context)
        fallback_note = spin_fallback.note()
        if fallback_note is not None:
            full_grid_notes.append(fallback_note)
    if (
        symmetry_mode == "on"
        and orbital_decomposition is None
        and not (
            selection.hartree_backend == "native"
            and hartree_reduction is not None
        )
    ):
        raise ValueError(
            "symmetry='on' requested, but the selected backend has no usable "
            "symmetry-reduced component for this calculation"
        )

    boundary_method = reference.input.hartree.boundary_method
    boundary_setup_eligible = (
        selection.hartree_backend == "native"
        and reference.grid.settings.domain_shape == "sphere"
        and boundary_method in {"auto", "multipole"}
    )
    boundary_overlap_enabled = os.environ.get(
        "PARSEC_OVERLAP_HARTREE_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    # The worker is useful when there is independent GPU setup to hide it
    # behind.  Native/SciPy-only paths keep deterministic inline construction
    # and avoid paying thread-pool overhead for no overlap opportunity.
    boundary_setup_future = None
    boundary_setup_executor = None
    boundary_setup_status = "not applicable"
    boundary_setup_seconds = 0.0
    boundary_setup_wait_seconds = 0.0
    boundary_setup_overlapped_seconds = 0.0
    boundary_setup_reduction = hartree_reduction
    # The Poisson CG arrays, the GPU boundary geometry and the resident
    # Hartree maps exchange device arrays and therefore share one device.  It
    # is fixed here because the boundary worker starts before the sector
    # eigensolver exists.  ``None`` leaves every object on its default device.
    # A sector worker builds none of them and has nothing to place.
    # One process with several GPUs places its sectors by size (PARSEC_CUPY_SECTOR_PLACEMENT), as the MPI runner
    # places those of one rank: decided here, before the Hartree device that follows from it and the eigensolver
    # that takes it.  None keeps the cyclic deal of before.
    process_sector_groups = process_divided_bases = process_placement_note = None
    # What the input fixes of the filter work: asked once, for the placement and the sector eigensolver.
    sector_filter_plan = (
        None
        if orbital_decomposition is None
        else share_filter_plan
        if share_filter_plan is not _NOT_PLANNED
        else _filter_plan(reference)
    )
    if orbital_decomposition is not None and mpi_context is None and selection.selected == "cupy":
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        process_sector_groups, process_divided_bases, process_placement_note = _process_sector_placement(
            reference, orbital_decomposition, selected_device_ids(int(cp.cuda.Device().id)), sector_filter_plan
        )
    hartree_device = None
    hartree_device_request = _hartree_device_request()
    # ``auto`` asks for a device only where one of the three is built on a
    # GPU: the CG, or the boundary geometry, whose own ``auto`` follows the
    # CG.  An index is checked against the devices of the process even so.
    hartree_on_gpu = hartree_linear_backend == "cupy" or (
        os.environ.get("PARSEC_HARTREE_BOUNDARY_BACKEND", "native").strip().lower()
        == "cupy"
    )
    if (
        hartree_device_request is not None
        and (hartree_device_request != "auto" or hartree_on_gpu)
        and selection.selected == "cupy"
        and not sector_worker
    ):
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        current_device = int(cp.cuda.Device().id)
        process_devices = selected_device_ids(current_device)
        hartree_device = _select_hartree_device(
            hartree_device_request,
            process_devices,
            (
                # One full-grid operator on the current device.
                {0: (current_device,)}
                if orbital_decomposition is None
                # The groups of one process placed by size.
                else process_sector_groups
                if process_sector_groups is not None
                else _planned_sector_device_groups(
                    orbital_decomposition.representation_count,
                    process_devices,
                    mpi_context,
                    costs=(
                        None
                        if mpi_context is None
                        else _sector_placement_costs(orbital_decomposition, _filter_plan(reference))
                    ),
                    divisible=(
                        None
                        if mpi_context is None
                        else _sector_bases_divisible(
                            orbital_decomposition, _filter_plan(reference), process_devices, mpi_context,
                            _sector_placement_costs(orbital_decomposition, _filter_plan(reference)),
                        )
                    ),
                    unit_devices=(
                        _unit_device_ids(orbital_decomposition, _filter_plan(reference), process_devices)
                        if mpi_context is None
                        else None
                    ),
                )
            ),
        )
    if (
        hartree_linear_backend == "cupy"
        and not hartree_linear_named
        and orbital_decomposition is None
        and not sector_worker
    ):
        # A full-grid eigensolver has no memory rule that plans around the
        # GPU Hartree arrays: ``auto`` keeps them on the GPU only where they
        # take a small share of what the device has free (decided before the
        # boundary thread, which follows the CG).
        from .Hartree.gpu_memory import (
            FREE_SHARE_BESIDE_FULL_GRID,
            device_free_bytes,
            planned_gpu_hartree_bytes,
        )

        planned_hartree_bytes = planned_gpu_hartree_bytes(
            reference.grid.size if hartree_reduction is None else hartree_reduction.wedge_size,
            reference.grid.settings.stencil_half_width,
            gpu_boundary=hartree_reduction is None,
        )
        free_hartree_bytes = device_free_bytes(hartree_device)
        if (
            free_hartree_bytes is not None
            and planned_hartree_bytes > FREE_SHARE_BESIDE_FULL_GRID * free_hartree_bytes
        ):
            hartree_linear_backend = "native"
            hartree_linear_reason = (
                "auto: the host CG, the GPU Hartree arrays "
                f"({planned_hartree_bytes / 2**20:.0f} MiB) would take more than "
                f"1/{round(1 / FREE_SHARE_BESIDE_FULL_GRID)} of the "
                f"{free_hartree_bytes / 2**20:.0f} MiB free on their device beside "
                "a full-grid eigensolver, which does not plan around them"
            )
    # PARSEC_CUPY_RESIDENT_HARTREE: where the GPU CG solves the wedge of the
    # SCF fields, the boundary of that wedge is built on its device too and
    # every solve stays there (decided here, before the boundary thread).
    resident_request = _resident_hartree_request()
    resident_wedge, resident_reason = _resident_hartree_plan(
        resident_request,
        linear_backend=hartree_linear_backend,
        hartree_reduction=hartree_reduction,
        orbital_decomposition=orbital_decomposition,
        # (the wedge of the SCF fields: the orbits of the full group where they take them)
        orbital_reduction=orbital_reduction if field_reduction is None else field_reduction,
        boundary_builder_planned=boundary_setup_eligible and selection.selected == "cupy",
    )
    resident_wedge = resident_wedge and not sector_worker
    planned_resident_reason = resident_reason
    resident_prefix = "auto" if resident_request == "auto" else "PARSEC_CUPY_RESIDENT_HARTREE=1"
    boundary_setup_notes: dict = {}
    # The boundary builder of the chain lies on the device of the GPU CG: the
    # Hartree device, or without one (a single device, or
    # PARSEC_HARTREE_DEVICE=off) the first device of the sectors, which the
    # CG takes; the default device of the boundary thread may be another.
    boundary_device = hartree_device
    if resident_wedge and boundary_device is None:
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        boundary_device = selected_device_ids(int(cp.cuda.Device().id))[0]
    if (
        boundary_setup_eligible
        and boundary_overlap_enabled
        and selection.selected == "cupy"
        and not sector_worker
    ):
        # The thread uploads the geometry, compiles and launches kernels,
        # the one of the atomic tail among them, and synchronizes its own
        # stream.  It synchronizes no device and takes no cuBLAS or cuSOLVER
        # handle, so the graph capture of the Poisson solver, which may be
        # open on the same device before the join, stays valid while the
        # thread works and when it ends (``backends/cupy_capture.py``).
        boundary_setup_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-hartree-setup",
        )
        # A preparation that fails before the join below waits for the
        # thread on its way out.
        preparation.join_on_exit(
            lambda: boundary_setup_executor.shutdown(wait=True)
        )
        boundary_setup_future = boundary_setup_executor.submit(
            stages.timed("hartree_boundary_setup", _build_native_boundary_builder),
            reference,
            hartree_reduction,
            symmetry_cache_directory=symmetry_cache_directory,
            symmetry_geometry_cache_info=symmetry_geometry_cache_info,
            device_id=boundary_device,
            tail_on_device=True,
            gpu_kernels=True,
            gpu_linear=hartree_linear_backend == "cupy",
            resident_wedge=resident_wedge,
            notes=boundary_setup_notes,
        )
        boundary_setup_status = "overlapped with GPU orbital setup"

    if mpi_context is not None and orbital_decomposition is None:
        raise ValueError(
            "MPI SCF requires a CuPy orbital symmetry decomposition: the ranks divide "
            "the symmetry sectors of the structure, and this one has no supported "
            "point-group symmetry (or symmetry is switched off). Run it on one node "
            "with --serial-control, which uses all GPUs of that node"
        )

    # The full-grid GPU operator serves only the pure-CuPy Poisson solver,
    # which a sector worker does not build.
    defer_full_gpu_operator = orbital_decomposition is not None and (
        selection.hartree_backend != "cupy" or sector_worker
    )
    # The KB factor with the LDA+U columns appended on CuPy (None: the KB
    # factor of the reference system).
    from .Hamiltonian.hubbard_device import device_hubbard_for, orbital_nonlocal_operator

    orbital_nonlocal = orbital_nonlocal_operator(reference, selection.selected)
    implementation = stages.timed("backend_build", _build_backend)(
        reference,
        selection,
        defer_cupy_device_operator=defer_full_gpu_operator,
        nonlocal_operator=orbital_nonlocal,
    )
    native_kernel_names: tuple[str, ...] = ()
    if (
        selection.finite_difference_builder == "native"
        or selection.hartree_backend == "native"
    ):
        from .backends.native import native_build_info

        native_kernel_names = tuple(
            str(name)
            for name in native_build_info().get("implemented_kernels", ())
        )
    native_boundary_builder = None
    native_boundary_cache_info = None
    native_symmetry_boundary = False
    hartree_boundary_tail = None
    resident_hartree = None
    # The solver of A u = rho of PARSEC_HARTREE_VARIATIONAL, set by the route of the Hartree solve.
    zero_boundary_solver = None
    # The GPU direct boundary of an ellipsoid or a cylinder, where one is built.
    direct_boundary = None
    native_xc_evaluator = None
    scf_reducer = None
    symmetry_eigensolver = None
    if orbital_decomposition is not None:
        from .Eigensolvers import CuPySymmetrySCFEigensolver
        from .Eigensolvers.symmetry import irrep_units_refusal

        # The units of irreducible representations (PARSEC_ORBITAL_IRREPS) need a CHEBFF or CHEBDAV first solve:
        # another one is refused here, before the SCF, for the first solver the input names.
        input_eigensolver = getattr(getattr(reference, "input", None), "eigensolver", None)
        if input_eigensolver is not None:
            from types import SimpleNamespace

            refusal = irrep_units_refusal(
                orbital_decomposition, SimpleNamespace(initial_method=getattr(input_eigensolver, "method", "chebff")))
            if refusal is not None:
                raise ValueError(refusal)
        stage_started = perf_counter()
        # An oblique cell (PARSEC_OBLIQUE_SECTORS=auto): the Laplacian of its grid with the weights of the second
        # differences that its admissible operations exchange averaged, where a cell typed to finite precision leaves
        # them unequal by about 1e-11 (Symmetry/oblique_frame.sector_negative_laplacian); its own otherwise.
        sector_kinetic = None
        if symmetry_frame is not None:
            from .Symmetry.oblique_frame import sector_negative_laplacian

            sector_kinetic = stages.timed("sector_kinetic", sector_negative_laplacian)(
                reference.grid, symmetry_frame.operations
            )
        symmetry_eigensolver = CuPySymmetrySCFEigensolver(
            implementation.eigensolver_operator,
            reference.negative_laplacian if sector_kinetic is None else sector_kinetic,
            reference.nonlocal_operator if orbital_nonlocal is None else orbital_nonlocal,
            orbital_decomposition,
            timing_stats=implementation.timing_stats,
            local_potential_getter=lambda: implementation.local_potential,
            operator_cache_directory=(
                None
                if symmetry_cache_directory is None
                else Path(symmetry_cache_directory)
            ),
            # The key of a deferred Laplacian if a cache directory made it
            # hash one.  Without it the reduced-operator cache asks the
            # descriptor itself, and only when it looks a key up.
            kinetic_cache_key=(
                getattr(reference.negative_laplacian, "hashed_cache_key", None) if sector_kinetic is None else None
            ),
            decomposition_cache_key=(
                None
                if symmetry_representation_cache_info is None
                else symmetry_representation_cache_info.key
            ),
            **({"mpi_context": mpi_context} if mpi_context is not None else {}),
            # The filter work that the tiles of a sector stencil must repay
            # their packing with.
            filter_plan=sector_filter_plan,
            # the shared full grid of a spin fallback: the GPUs of the process share the sector of the identity
            **({"process_device_group": True} if spin_identity_sector and mpi_context is None else {}),
            # One process placed by size (PARSEC_CUPY_SECTOR_PLACEMENT); none keeps the cyclic deal (the sector of
            # the identity of a spin fallback then takes the GPUs of the process all the same, above).
            **(
                {"sector_device_groups": process_sector_groups, "divided_bases": process_divided_bases}
                if process_sector_groups is not None
                else {}
            ),
        )
        implementation.symmetry_eigensolver = symmetry_eigensolver
        implementation.eigenproblem_solver = symmetry_eigensolver
        # Fix_Number_Eigenvalues under PARSEC_FIXED_STATES_BUFFER=off: the sectors keep the counts of the first
        # solve (auto: their own counts with the buffer, as a run without the keyword, and a NOTE says so).
        from parsec_python.models import fixed_states_note, fixed_states_sector_buffer

        from .Eigensolvers.symmetry import fixed_state_count_requested

        symmetry_eigensolver.fixed_state_count = fixed_state_count_requested(reference)
        # Fermi occupations (not occup.in): the level that the window ends in may share the states left of it
        # (PARSEC_WEIGHTED_STATE_COUNT, CuPySymmetrySCFEigensolver._weighted_selection)
        symmetry_eigensolver.fermi_occupations = (
            getattr(getattr(getattr(reference, "input", None), "scf", None), "fixed_occupations", None) is None
        )
        _fixed_eigensolver = getattr(getattr(reference, "input", None), "eigensolver", None)
        # (whose states within the eigensolver tolerance of the last count as one level: weighted_level_width)
        symmetry_eigensolver.level_tolerance = getattr(_fixed_eigensolver, "tolerance", None)
        symmetry_eigensolver.fixed_state_buffer = fixed_states_sector_buffer(_fixed_eigensolver)
        _fixed_note = fixed_states_note(_fixed_eigensolver)
        if _fixed_note is not None and not sector_worker:
            full_grid_notes.append(_fixed_note)
        # A sector that shares its basis gives the device with the Hartree
        # objects work beside its own only where it has no other device for
        # it and the sharing rule counts the room for it there.
        symmetry_eigensolver.hartree_device = hartree_device
        # Density and every local scalar potential are totally symmetric.
        # Retain one physical value per orbit for Anderson history, residual
        # norms, and energy quadrature; expand only the mixed field required
        # by the existing Hamiltonian/Hartree interfaces.
        from .SCF import SymmetrySCFReducer
        from .SCF.symmetry_fields import group_average_applies

        # Weighted sectors of a subgroup H: the full-grid inputs (ionic, core and start fields) are averaged over
        # the full group as well (PARSEC_SECTOR_INPUT_GROUP_AVERAGE, read at the start of the preparation).
        # The orbits of the full group where the fields take them (field_reduction, above): there the orbit
        # average of a full-grid input is its average over that group already.
        scf_reducer = SymmetrySCFReducer(
            orbital_reduction if field_reduction is None else field_reduction,
            group_average_inputs=(
                field_reduction is None
                and input_group_average == "auto"
                and group_average_applies(orbital_reduction)
            ),
        )
        from .Occupations import CuPySymmetryDensityBuilder

        implementation.orbital_density_builder = CuPySymmetryDensityBuilder(
            implementation.orbital_density_builder,
            implementation.timing_stats,
            scf_reducer,
        )
        implementation.statistics.initialization_seconds = (
            implementation.timing_stats.initialization_seconds
        )
        stages.add("sector_eigensolver", stage_started)

    if ionic_fields is not None and ionic_fields.started:
        # The XC evaluator below is the first reader of an ionic field, and
        # the Poisson solver after it captures a graph: the thread has ended
        # before either.
        reference = ionic_fields.result()
        if ionic_fields.devices:
            preparation_overlap["ionic_sum_devices"] = ionic_fields.devices
    if is_periodic and spin_polarized and not sector_worker:
        # The starting densities of the two channels that run_spin_scf reads (initchrg.f90 with the lattice
        # images), which the periodic preparation leaves out (SCF/gamma_spin.py).
        from parsec_python.SCF.pbc import with_initial_spin_densities

        reference = with_initial_spin_densities(reference)

    stage_started = perf_counter()
    if sector_worker:
        # The frozen core density was not built on this rank.
        native_xc_evaluator = _root_rank_only("the XC evaluator")
    elif (
        selection.hartree_backend == "native"
        and xc_functional == "ca"
    ):
        from .V_xc import NativeCALDAEvaluator

        if "CALDAEvaluator" in native_kernel_names:
            native_xc_evaluator = NativeCALDAEvaluator(
                reference.core_density,
                reference.grid.volume_element,
                scf_reducer,
            )

    spin_xc_evaluator = None
    # A hybrid functional: the semilocal part of the hybrid and the start
    # functional of its Fock loop (V_xc/device_hybrid_xc.py, device_xc.py).
    hybrid_start_xc = (None, None)
    from .V_xc.device_xc import nuclear_source

    if native_xc_evaluator is None and getattr(
        getattr(getattr(reference, "input", None), "scf", None), "is_meta_gga", False
    ):
        # A meta-GGA: the functional of the reference on the host, the fields
        # of the sectors expanded for it (V_xc/meta_gga_xc.py).
        # On the device where the evaluator runs on CuPy arrays (PARSEC_CUPY_XC
        # not off, a cluster or orthorhombic grid, no Laplacian dependence).
        from parsec_python.V_xc.meta_gga import meta_gga_evaluator

        from .V_xc import device_xc_mode
        from .V_xc.meta_gga_xc import (
            DeviceMetaGGAOnOrbits,
            HostMetaGGAOnOrbits,
            HostMetaGGASpinOnOrbits,
        )

        native_xc_evaluator = HostMetaGGAOnOrbits(lambda: reference, scf_reducer)
        spin_xc_evaluator = HostMetaGGASpinOnOrbits(
            lambda: reference, scf_reducer if spin_sectors else None
        )
        if selection.selected == "cupy" and device_xc_mode() != "off" and not sector_worker:
            functional = meta_gga_evaluator(reference.input.scf)
            # The fields of weighted sectors (PARSEC_ORBITAL_GROUP_META_GGA): the device stencil with the
            # operation table of their group, which exchanges axes.
            from .Hamiltonian.meta_gga_device import weighted_sectors

            permutations = weighted_sectors(getattr(symmetry_eigensolver, "decomposition", None))
            try:
                if spin_polarized:
                    spin_xc_evaluator = DeviceMetaGGAOnOrbits(
                        functional,
                        reference.grid,
                        reducer=scf_reducer if spin_sectors else None,
                        device_id=hartree_device if isinstance(hartree_device, int) else 0,
                        core_density=lambda: reference.core_density,
                        spin=True,
                        permutations=permutations and spin_sectors,
                        nuclear_gradient=nuclear_source(lambda: reference),
                    )
                else:
                    native_xc_evaluator = DeviceMetaGGAOnOrbits(
                        functional,
                        reference.grid,
                        reducer=scf_reducer,
                        device_id=hartree_device if isinstance(hartree_device, int) else 0,
                        core_density=lambda: reference.core_density,
                        permutations=permutations,
                        nuclear_gradient=nuclear_source(lambda: reference),
                    )
            except ValueError:
                # No device evaluation of this functional or grid: the host one.
                pass
            planned = getattr(native_xc_evaluator, "planned_device_bytes", None)
            if symmetry_eigensolver is not None and planned is not None and not spin_polarized:
                # The memory rules of the sectors count what the device
                # evaluator keeps and takes on its device (the spin one below).
                symmetry_eigensolver.resident_device_bytes = {
                    int(native_xc_evaluator.device_id): int(planned)
                }
        if getattr(reference.input.scf, "is_hybrid", False) and not sector_worker:
            # A meta-GGA hybrid: the start functional (PBE) of its Fock loop where
            # PARSEC_CUPY_XC puts it on the device; the semilocal part is the meta-GGA's.
            hybrid_start_xc = _hybrid_xc_evaluators(
                reference,
                backend=selection.selected,
                reducer=scf_reducer,
                device_id=hartree_device,
                core_density=lambda: reference.core_density,
                nuclear_gradient=nuclear_source(lambda: reference),
            )[2]
    elif native_xc_evaluator is None and getattr(
        getattr(getattr(reference, "input", None), "scf", None), "is_hybrid", False
    ):
        native_xc_evaluator, spin_xc_evaluator, hybrid_start_xc = _hybrid_xc_evaluators(
            reference,
            backend=selection.selected,
            reducer=scf_reducer,
            device_id=hartree_device,
            core_density=lambda: reference.core_density,
            nuclear_gradient=nuclear_source(lambda: reference),
        )
    elif native_xc_evaluator is None:
        # The other functionals on the device where PARSEC_CUPY_XC puts them
        # there (V_xc/device_xc.py); None leaves them to the host below.
        from .V_xc import accelerated_xc_evaluators

        native_xc_evaluator, spin_xc_evaluator = accelerated_xc_evaluators(
            reference,
            backend=selection.selected,
            reducer=scf_reducer,
            device_id=hartree_device,
            # The structure of the present atoms (move_atoms rebinds it).
            core_density=lambda: reference.core_density,
            nuclear_gradient=nuclear_source(lambda: reference),
        )
        planned = getattr(native_xc_evaluator, "planned_device_bytes", None)
        if symmetry_eigensolver is not None and planned is not None:
            # The memory rules of the sectors count what the device
            # evaluator keeps and takes on its device.
            symmetry_eigensolver.resident_device_bytes = {
                int(native_xc_evaluator.device_id): int(planned)
            }

    if scf_reducer is not None and native_xc_evaluator is None:
        # The readable XC evaluators accept complete Cartesian arrays.  PBE
        # specifically needs that layout because its density gradient couples
        # neighboring symmetry orbits; CA-LDA is pointwise but uses the same
        # public array contract.  Expand the invariant density for this narrow
        # operation, then retain one value per orbit again for the rest of the
        # accelerated SCF.  This is an exact representation adapter, not a
        # different functional or quadrature.
        from parsec_python.V_xc import XCResult

        def symmetry_xc_evaluator(density):
            evaluated = reference.evaluate_xc(scf_reducer.to_full(density))
            return XCResult(
                potential=scf_reducer.from_full(evaluated.potential),
                energy_per_electron=scf_reducer.from_full(
                    evaluated.energy_per_electron
                ),
                energy_density=scf_reducer.from_full(evaluated.energy_density),
                total_energy=evaluated.total_energy,
            )

        native_xc_evaluator = symmetry_xc_evaluator
        if (
            not is_periodic
            and not spin_polarized
            and not sector_worker
            and xc_functional == "ca"
            and selection.selected == "cupy"
            and selection.hartree_backend != "native"
        ):
            # The CA-LDA kernel on the orbits is the native extension's (above).
            full_grid_notes.append(_cluster_xc_full_grid_note(selection, functional="CA-LDA"))
    if spin_sectors and spin_xc_evaluator is None:
        # A spin functional the device does not evaluate, on the orbit fields
        # of the sectors: expanded, evaluated on the host, reduced (exact for
        # invariant channels).  The structure of the present atoms.
        from .SCF.spin_sectors import HostSpinXCOnOrbits

        spin_xc_evaluator = HostSpinXCOnOrbits(
            lambda up, down: reference.evaluate_spin_xc(up, down),
            scf_reducer,
            _readable_spin_xc_label(xc_functional),
        )
    if spin_sectors and symmetry_eigensolver is not None and not getattr(
        getattr(getattr(reference, "input", None), "scf", None), "is_hybrid", False
    ):
        planned = getattr(spin_xc_evaluator, "planned_device_bytes", None)
        if planned is not None:
            # The memory rules of the sectors count the device spin evaluator
            # (a hybrid's are counted with its exchange below).
            symmetry_eigensolver.resident_device_bytes = {int(spin_xc_evaluator.device_id): int(planned)}
    stages.add("xc_evaluator", stage_started)

    # The Hartree solver of the selected backend, with its boundary
    # geometry and, where they are asked for, the resident maps.
    hartree_started = perf_counter()
    if sector_worker:
        # No Poisson solver, boundary geometry or resident Hartree maps:
        # the root rank solves Hartree and sends the effective potential.
        accelerated_hartree = _root_rank_only("the Hartree solver")

    elif is_periodic:
        # solve_scipy_hartree/CuPyPoissonSolver/NativePoissonSolver all build
        # an isolated-cluster boundary (multipole or direct Coulomb); none
        # apply to a periodic cell. Delegate straight to the already-
        # validated periodic CG solve (SCF.pbc.solve_periodic_hartree via
        # reference.solve_hartree) -- _selection_for_problem forces
        # hartree_backend to "scipy" for periodic precisely so none of the
        # branches below this one can be reached instead.
        # In one sector of the identity (periodic_sector_refusal) the SCF
        # keeps its fields as orbit fields of the identity, one value per grid
        # point in the order of the grid: the full-grid solve sees them
        # expanded, and its potentials go back to orbit fields, both exact
        # copies.
        def accelerated_hartree(density, initial_potential=None, **kwargs):
            if scf_reducer is None:
                return reference.solve_hartree(density, initial_potential, **kwargs)
            result = reference.solve_hartree(
                scf_reducer.to_full(density),
                None if initial_potential is None else scf_reducer.to_full(initial_potential),
                **kwargs,
            )
            return replace(
                result,
                potential=scf_reducer.from_full(result.potential),
                right_hand_side=scf_reducer.from_full(result.right_hand_side),
            )

    elif selection.hartree_backend == "cupy":
        from .Hartree.cupy_poisson import CuPyPoissonSolver

        if selection.selected != "cupy":
            raise RuntimeError("CuPy Hartree requires the CuPy execution backend")
        poisson_solver = CuPyPoissonSolver(implementation.device_operator)
        implementation.poisson_solver = poisson_solver
        from parsec_python.Hartree.variational import ZeroBoundaryDeviceSolver

        zero_boundary_solver = ZeroBoundaryDeviceSolver(poisson_solver.negative_laplacian, reference.grid)
        hartree_boundary_tail = _hartree_boundary_tail(
            reference, on_device=not tail_on_host, device_id=None
        )

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            # ``CuPyPoissonSolver`` owns the full-grid finite-difference
            # operator and boundary builder.  The symmetry-aware nonlinear
            # SCF path, however, keeps scalar fields as one physical value per
            # orbit.  Expand only across this full-grid solver boundary, then
            # immediately compact the returned invariant fields again.  This
            # is an exact representation conversion; no density, boundary
            # condition, or Poisson tolerance is changed.
            full_density = (
                density
                if scf_reducer is None
                else scf_reducer.to_full(density)
            )
            full_initial = (
                initial_potential
                if scf_reducer is None or initial_potential is None
                else scf_reducer.to_full(initial_potential)
            )
            result = poisson_solver.solve(
                full_density,
                reference.grid,
                reference.input.hartree,
                full_initial,
                boundary_tail=hartree_boundary_tail,
                **kwargs,
            )
            if scf_reducer is None:
                return result
            return replace(
                result,
                potential=scf_reducer.from_full(result.potential),
                right_hand_side=scf_reducer.from_full(
                    result.right_hand_side
                ),
            )

    elif selection.hartree_backend == "native":
        from .Hartree.native_poisson import NativePoissonSolver
        from .Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
        from .Laplacian import (
            DeferredNativeNegativeLaplacian,
            materialize_negative_laplacian,
        )

        def concrete_negative_laplacian(operator):
            return (
                materialize_negative_laplacian(operator)
                if isinstance(operator, DeferredNativeNegativeLaplacian)
                else operator
            )

        from .Hartree.gpu_memory import cuda_allocation_errors

        def build_poisson_solver(linear_backend, reduction):
            """The Poisson solver of ``linear_backend`` and the reduction it uses."""

            poisson_factory = NativePoissonSolver
            if linear_backend == "cupy":
                from functools import partial
                from .Hartree.cupy_prepared import CuPyPreparedPoissonSolver

                poisson_factory = partial(
                    CuPyPreparedPoissonSolver,
                    device_id=(
                        hartree_device
                        if hartree_device is not None
                        else symmetry_eigensolver.device_ids[0]
                        if symmetry_eigensolver is not None else None
                    ),
                    # A large wedge takes the CG to every device of the process
                    # (PARSEC_HARTREE_CG_DEVICES), the Hartree device first.
                    device_ids=(
                        tuple(symmetry_eigensolver.device_ids)
                        if symmetry_eigensolver is not None else None
                    ),
                )
            if reduction is None:
                return poisson_factory(
                    concrete_negative_laplacian(reference.negative_laplacian)
                ), None
            try:
                reusable_reduced_operator = (
                    symmetry_eigensolver is not None
                    and orbital_reduction is reduction
                )
                if field_reduction is not None and reduction is field_reduction:
                    # The orbits of the full group (PARSEC_SECTOR_FIELD_GROUP=full): their operator from the stencil
                    # of the grid, without the full-grid matrix; the packed form for the GPU CG where its palette
                    # holds the coefficients, CSR otherwise and for the host CG.
                    from .Symmetry.field_orbits import field_negative_laplacian

                    reduced_operator = stages.timed("field_laplacian", field_negative_laplacian)(
                        reference.grid, reduction
                    )
                    if linear_backend != "cupy" and hasattr(reduced_operator, "to_csr"):
                        reduced_operator = reduced_operator.to_csr()
                    reusable_reduced_operator = True
                elif not reusable_reduced_operator:
                    reduced_operator = concrete_negative_laplacian(
                        reference.negative_laplacian
                    )
                elif linear_backend == "cupy":
                    # The GPU CG reads the slot-major stencil the totally
                    # symmetric sector was packed into.  Converting it to CSR
                    # and packing that again would put the same neighbor and
                    # the same coefficient into every slot.
                    reduced_operator = (
                        symmetry_eigensolver.totally_symmetric_stencil
                    )
                else:
                    # The native CG copies CSR buffers, formed here on demand.
                    reduced_operator = (
                        symmetry_eigensolver.totally_symmetric_negative_laplacian
                    )
                return SymmetryReducedPoissonSolver(
                    reduced_operator,
                    reduction,
                    operator_is_reduced=reusable_reduced_operator,
                    solver_factory=poisson_factory,
                ), reduction
            except (ValueError, RuntimeError) as error:
                # A GPU CG of ``auto`` that fails on the wedge gives way to the
                # host CG on the same wedge (below), not to a GPU CG of the
                # full grid |G| times its size.  An unusable reduction leaves
                # the full grid, as for the host CG; the GPU CG asked for by
                # name stops instead.
                if linear_backend == "cupy" and not hartree_linear_named:
                    raise _GPUHartreeUnavailable(str(error)) from error
                if (symmetry_mode == "on" or legacy_hartree_forced
                        or (linear_backend == "cupy" and hartree_linear_named)):
                    raise
                return poisson_factory(
                    concrete_negative_laplacian(reference.negative_laplacian)
                ), None

        try:
            poisson_solver, hartree_reduction = build_poisson_solver(
                hartree_linear_backend, hartree_reduction
            )
        except (_GPUHartreeUnavailable, *cuda_allocation_errors()) as error:
            # CuPy's OutOfMemoryError is a MemoryError, and an exhausted
            # device also answers through CuPy's CUDA runtime and driver
            # errors.  The CG chosen by ``auto`` gives way to the host CG on
            # the same reduction; one asked for by name does not.
            if hartree_linear_backend != "cupy" or hartree_linear_named:
                raise
            cause = error.__cause__ if isinstance(error, _GPUHartreeUnavailable) else error
            hartree_linear_backend = "native"
            hartree_linear_reason = (
                "auto: the host CG, the GPU CG could not be built "
                f"({type(cause).__name__}: {cause})"
            )
            poisson_solver, hartree_reduction = build_poisson_solver(
                "native", hartree_reduction
            )
        stages.add("poisson_solver", hartree_started)
        from parsec_python.Hartree.variational import ZeroBoundaryBackendSolver

        # The CG of this route (host or GPU, full grid or wedge) solves A u = rho
        # of PARSEC_HARTREE_VARIATIONAL; the resident chain shares it.  The
        # wedge solver (SymmetryReducedPoissonSolver) comes with its reduction
        # from build_poisson_solver and wraps the CG of the reduced operator; a
        # full-grid solver comes with none.
        zero_boundary_solver = (
            ZeroBoundaryBackendSolver(poisson_solver.solver.backend, reference.grid, hartree_reduction)
            if hartree_reduction is not None
            else ZeroBoundaryBackendSolver(poisson_solver.backend, reference.grid)
        )
        if boundary_setup_eligible:
            if boundary_setup_future is not None:
                wait_started = perf_counter()
                try:
                    (
                        native_boundary_builder,
                        native_symmetry_boundary,
                        native_boundary_cache_info,
                        boundary_setup_seconds,
                    ) = boundary_setup_future.result()
                finally:
                    boundary_setup_wait_seconds = perf_counter() - wait_started
                    boundary_setup_executor.shutdown(wait=True)
                    stages.add("hartree_boundary_setup_wait", wait_started)
                boundary_setup_overlapped_seconds = max(
                    0.0,
                    boundary_setup_seconds - boundary_setup_wait_seconds,
                )
                # A rare reduced-Poisson construction failure changes the
                # required boundary representation after the worker began.
                # Discard that speculative symmetry builder and reconstruct
                # the exact full-grid fallback rather than mixing layouts.
                if (
                    boundary_setup_reduction is not None
                    and hartree_reduction is None
                    and native_symmetry_boundary
                ):
                    (
                        native_boundary_builder,
                        native_symmetry_boundary,
                        native_boundary_cache_info,
                        rebuild_seconds,
                    ) = stages.timed(
                        "hartree_boundary_setup", _build_native_boundary_builder
                    )(
                        reference,
                        None,
                        symmetry_cache_directory=symmetry_cache_directory,
                        symmetry_geometry_cache_info=(
                            symmetry_geometry_cache_info
                        ),
                        device_id=hartree_device,
                        # The tail in rows of the discarded builder, or a
                        # new one from the evaluator that builder was given.
                        boundary_tail=_attached_boundary_tail(
                            native_boundary_builder
                        ),
                        tail_on_device=selection.selected == "cupy",
                        gpu_kernels=selection.selected == "cupy",
                        gpu_linear=hartree_linear_backend == "cupy",
                    )
                    boundary_setup_seconds += rebuild_seconds
                    boundary_setup_status = (
                        "overlap discarded after reduced-Poisson fallback"
                    )
            else:
                (
                    native_boundary_builder,
                    native_symmetry_boundary,
                    native_boundary_cache_info,
                    boundary_setup_seconds,
                ) = stages.timed(
                    "hartree_boundary_setup", _build_native_boundary_builder
                )(
                    reference,
                    hartree_reduction,
                    symmetry_cache_directory=symmetry_cache_directory,
                    symmetry_geometry_cache_info=symmetry_geometry_cache_info,
                    device_id=boundary_device,
                    tail_on_device=selection.selected == "cupy",
                    gpu_kernels=selection.selected == "cupy",
                    gpu_linear=hartree_linear_backend == "cupy",
                    resident_wedge=resident_wedge,
                    notes=boundary_setup_notes,
                )
                boundary_setup_status = (
                    "inline (overlap disabled)"
                    if selection.selected == "cupy"
                    else "inline (no independent GPU setup)"
                )

        def host_boundary(
            status: str, *, gpu_kernels: bool, gpu_linear: bool, tail_on_device: bool = True
        ) -> None:
            """Put the boundary builder of the route before the chain in its place.

            ``gpu_kernels`` and ``gpu_linear`` are those that route asked
            the builder with: the orbit table where it fits, and otherwise
            the GPU kernels or the host recurrences as they decide.  A tail
            in rows comes from the device kernel if ``tail_on_device``.
            """

            nonlocal native_boundary_builder, native_symmetry_boundary, hartree_boundary_tail
            nonlocal native_boundary_cache_info, boundary_setup_seconds, boundary_setup_status
            discarded = native_boundary_builder
            native_boundary_builder = None
            (
                native_boundary_builder,
                native_symmetry_boundary,
                native_boundary_cache_info,
                rebuild_seconds,
            ) = stages.timed("hartree_boundary_setup", _build_native_boundary_builder)(
                reference,
                hartree_reduction,
                symmetry_cache_directory=symmetry_cache_directory,
                symmetry_geometry_cache_info=symmetry_geometry_cache_info,
                device_id=hartree_device,
                boundary_tail=_attached_boundary_tail(discarded),
                tail_on_device=tail_on_device and selection.selected == "cupy",
                gpu_kernels=gpu_kernels,
                gpu_linear=gpu_linear,
            )
            del discarded
            implementation.native_boundary_builder = native_boundary_builder
            hartree_boundary_tail = _attached_boundary_tail(native_boundary_builder)
            boundary_setup_notes.pop("resident_boundary", None)
            boundary_setup_seconds += rebuild_seconds
            boundary_setup_status = status

        if (
            boundary_setup_notes.get("resident_boundary") == "gpu"
            and hartree_reduction is not None
            and hartree_linear_backend != "cupy"
        ):
            # The GPU CG gave way to the host CG: the boundary of the wedge as
            # that route asked for it before (the orbit table, or the GPU
            # kernels where it has none), with the host CG beside it.
            resident_reason = f"{resident_prefix}: the conjugate gradients run on the host"
            host_boundary(
                "rebuilt after the host-CG fallback",
                gpu_kernels=selection.selected == "cupy",
                gpu_linear=True,
            )
        elif boundary_setup_notes.get("resident_boundary", "gpu") != "gpu":
            resident_reason = f"{resident_prefix}: {boundary_setup_notes['resident_boundary']}"
        # CuPy's statistics bridge expects ``poisson_solver`` to expose CuPy
        # event timings.  Keep the native solver under a distinct, inspectable
        # name and accumulate its host timings directly below.
        implementation.native_poisson_solver = poisson_solver
        implementation.native_boundary_builder = native_boundary_builder
        hartree_boundary_tail = _attached_boundary_tail(native_boundary_builder)
        # The direct boundary sum of an ellipsoid or a cylinder, on a GPU.
        from .Hartree.cupy_direct_boundary import direct_boundary_for

        direct_boundary = (
            None
            if native_boundary_builder is not None
            else direct_boundary_for(
                reference.grid,
                reference.input.hartree,
                selected=selection.selected,
                device_id=hartree_device,
            )
        )
        implementation.direct_boundary = direct_boundary

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            total_started = perf_counter()
            rhs_started = perf_counter()
            # A symmetry-wedge calculation represents only the invariant
            # density.  Project before constructing multipoles so the
            # returned boundary object and corrected RHS describe the same
            # physical source, even if full-grid eigensolver roundoff left a
            # tiny difference between symmetry images.
            boundary_density = density
            if native_symmetry_boundary:
                right_hand_side, boundary = native_boundary_builder.build_reduced(
                    density
                )
            elif native_boundary_builder is None:
                from .SCF.symmetry_fields import SymmetryScalarField

                if isinstance(density, SymmetryScalarField):
                    # The orbit values of the SCF loop: the field they expand
                    # to is invariant already.
                    boundary_density = scf_reducer.to_full(density)
                elif hartree_reduction is not None:
                    boundary_density = hartree_reduction.project_invariant(density)
                if direct_boundary is not None:
                    right_hand_side, boundary = direct_boundary.build(
                        boundary_density,
                        reference.input.hartree,
                        hartree_boundary_tail,
                    )
                else:
                    right_hand_side, boundary = build_hartree_problem(
                        boundary_density,
                        reference.grid,
                        reference.input.hartree,
                        hartree_boundary_tail,
                    )
            else:
                # The size-adaptive full-grid C++ multipole builder can be
                # selected even when the nonlinear SCF stores an invariant
                # scalar field on the wedge.  Expand physical point values at
                # this narrow boundary only; the multipole/RHS equations and
                # all later reduced-Poisson algebra are unchanged.
                if scf_reducer is not None:
                    from .SCF.symmetry_fields import SymmetryScalarField

                    if isinstance(boundary_density, SymmetryScalarField):
                        boundary_density = scf_reducer.to_full(
                            boundary_density
                        )
                right_hand_side, boundary = native_boundary_builder.build(
                    boundary_density
                )
            rhs_seconds = perf_counter() - rhs_started
            solve_started = perf_counter()
            if native_symmetry_boundary:
                compact_hartree_output = (
                    scf_reducer is not None
                    and scf_reducer.reduction is hartree_reduction
                )
                native_result = poisson_solver.solve_reduced(
                    right_hand_side,
                    initial_potential,
                    reference.input.hartree,
                    return_wedge=compact_hartree_output,
                    **kwargs,
                )
                if scf_reducer is not None and not compact_hartree_output:
                    from dataclasses import replace as dataclass_replace

                    native_result = dataclass_replace(
                        native_result,
                        potential=scf_reducer.from_full(native_result.potential),
                        right_hand_side=scf_reducer.from_full(
                            native_result.right_hand_side
                        ),
                    )
            else:
                native_result = poisson_solver.solve(
                    right_hand_side,
                    initial_potential,
                    reference.input.hartree,
                    **kwargs,
                )
                if scf_reducer is not None and hartree_reduction is not None:
                    from dataclasses import replace as dataclass_replace
                    from .SCF.symmetry_fields import SymmetryScalarField

                    if not isinstance(
                        native_result.potential, SymmetryScalarField
                    ):
                        native_result = dataclass_replace(
                            native_result,
                            potential=scf_reducer.from_full(
                                native_result.potential
                            ),
                            right_hand_side=scf_reducer.from_full(
                                native_result.right_hand_side
                            ),
                        )
            solve_seconds = perf_counter() - solve_started
            implementation.statistics.hartree_solve_calls += 1
            implementation.statistics.hartree_rhs_seconds += rhs_seconds
            implementation.statistics.hartree_linear_solve_seconds += solve_seconds
            implementation.statistics.hartree_total_seconds += (
                perf_counter() - total_started
            )
            result = native_result.as_hartree_result(boundary)
            if hartree_boundary_tail is not None:
                result = replace(result, boundary_tail=hartree_boundary_tail)
            return result

        # The resident chain: the GPU boundary builder and the GPU CG on the
        # wedge of the SCF fields (PARSEC_CUPY_RESIDENT_HARTREE, auto by
        # default).  The host interface above stays behind it.
        resident_eligible = (
            type(native_boundary_builder).__name__ in _GPU_BOUNDARY_BUILDERS
            and hartree_linear_backend == "cupy"
            and hartree_reduction is not None
            and scf_reducer is not None
            and scf_reducer.reduction is hartree_reduction
            and (
                resident_request == "1"
                or int(hartree_reduction.full_size) <= np.iinfo(np.int32).max
            )
        )
        if resident_request == "1" and not resident_eligible:
            raise ValueError(
                "PARSEC_CUPY_RESIDENT_HARTREE=1 requires the GPU boundary and CG on the "
                f"symmetry wedge of the SCF fields ({resident_reason})"
            )
        resident_built_here = boundary_setup_notes.get("resident_boundary") == "gpu"
        if (
            resident_request != "0"
            and resident_eligible
            and int(native_boundary_builder.device_id)
            != int(poisson_solver.solver.backend.device_id)
        ):
            # Builder and CG on two devices would copy every solve across
            # them (and stop without peer access): the route before instead.
            message = (
                f"the boundary builder is on device {native_boundary_builder.device_id}, "
                f"the CG on device {poisson_solver.solver.backend.device_id}"
            )
            if resident_request == "1":
                raise ValueError(f"PARSEC_CUPY_RESIDENT_HARTREE=1: {message}")
            resident_eligible = False
            resident_reason = f"auto: {message}"
            if resident_built_here:
                host_boundary(
                    "rebuilt: builder and CG on two devices",
                    gpu_kernels=selection.selected == "cupy",
                    gpu_linear=True,
                )
        if resident_request != "0" and resident_eligible:
            from .Hartree.cupy_resident import CuPyResidentHartree

            try:
                resident_hartree = stages.timed("resident_hartree", CuPyResidentHartree)(
                    native_boundary_builder,
                    hartree_reduction,
                    poisson_solver.solver.backend,
                    reference.input.hartree,
                )
            except cuda_allocation_errors() as error:
                if resident_request == "1":
                    raise
                resident_hartree = None
                resident_reason = (
                    f"auto: the chain could not be built ({type(error).__name__}: {error})"
                )
                if resident_built_here:
                    host_boundary(
                        "rebuilt: the resident chain could not be built",
                        gpu_kernels=selection.selected == "cupy",
                        gpu_linear=True,
                    )
            else:
                if not resident_built_here:
                    # A GPU boundary named by PARSEC_HARTREE_BOUNDARY_BACKEND.
                    resident_reason = (
                        f"{resident_prefix}: the GPU boundary and CG on the symmetry wedge "
                        "of the SCF fields"
                    )

                def leave_chain():
                    """The boundary of the route before, on the host, after an allocation failure."""

                    host_boundary(
                        "rebuilt on the host after an allocation failure of the resident chain",
                        gpu_kernels=False,
                        gpu_linear=False,
                        tail_on_device=False,
                    )
                    return (
                        ("hartree_backend", "native-boundary/cupy-linear"),
                        ("hartree_boundary_kernel", _host_boundary_kernel(native_symmetry_boundary)),
                        ("hartree_boundary_device_bytes", "0"),
                        ("hartree_boundary_device", "host"),
                        ("hartree_boundary_setup", boundary_setup_status),
                    )

                accelerated_hartree = _ResidentHartreeSolve(
                    resident_hartree,
                    accelerated_hartree,
                    implementation.statistics,
                    # move_atoms rebinds the tail in rows.
                    tail=lambda: hartree_boundary_tail,
                    poisson_solver=poisson_solver,
                    named=resident_request == "1",
                    leave=leave_chain,
                )
        if resident_hartree is None and resident_reason == planned_resident_reason and resident_wedge:
            # Planned, but the Hartree solve changed at the join (a reduced
            # solver that could not be built leaves the full grid): the plan
            # again, with what was built.
            _, resident_reason = _resident_hartree_plan(
                resident_request,
                linear_backend=hartree_linear_backend,
                hartree_reduction=hartree_reduction,
                orbital_decomposition=orbital_decomposition,
                orbital_reduction=orbital_reduction,
                boundary_builder_planned=native_boundary_builder is not None,
            )
            if resident_reason == planned_resident_reason:
                resident_reason = f"{resident_prefix}: the boundary is built on the host"

        if symmetry_eigensolver is not None:
            # The memory rules of the sectors (state storage, shared bases)
            # count the GPU Hartree arrays on their device beside the XC
            # evaluator and the exchange of a hybrid.
            hartree_device_bytes = _gpu_hartree_device_bytes(
                poisson_solver, native_boundary_builder, resident_hartree
            )
            if hartree_device_bytes:
                counts = dict(symmetry_eigensolver.resident_device_bytes)
                for device, count in hartree_device_bytes.items():
                    counts[device] = counts.get(device, 0) + count
                symmetry_eigensolver.resident_device_bytes = counts

    elif selection.hartree_backend == "scipy":
        hartree_boundary_tail = _hartree_boundary_tail(
            reference, on_device=False, device_id=None
        )
        from parsec_python.Hartree.variational import ZeroBoundaryHostSolver

        zero_boundary_solver = ZeroBoundaryHostSolver(reference.negative_laplacian, reference.grid)

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            started = perf_counter()
            result = solve_scipy_hartree(
                density,
                reference.grid,
                reference.negative_laplacian,
                reference.input.hartree,
                initial_potential,
                boundary_tail=hartree_boundary_tail,
                **kwargs,
            )
            implementation.statistics.hartree_solve_calls += 1
            implementation.statistics.hartree_total_seconds += (
                perf_counter() - started
            )
            return result

    else:
        raise RuntimeError(
            f"unhandled Hartree backend {selection.hartree_backend!r}"
        )
    # PARSEC_HARTREE_VARIATIONAL: the derivative of the Hartree energy as the
    # potential of every solve (Hartree/variational.py), on the rank that solves.
    hartree_variational, hartree_variational_reason = _variational_hartree(
        reference,
        zero_boundary_solver,
        sector_worker=sector_worker,
        is_periodic=is_periodic,
        # the closures read the reference of the present structure (move_atoms rebinds it)
        charges=lambda: tail_charges(reference),
        atoms=lambda: reference.atoms,
    )
    if hartree_variational is not None:
        accelerated_hartree = VariationalHartreeSolve(accelerated_hartree, hartree_variational)
    stages.add("hartree_solver", hartree_started)

    boundary_on_gpu = type(native_boundary_builder).__name__ in _GPU_BOUNDARY_BUILDERS
    boundary_on_wedge_gpu = type(native_boundary_builder).__name__ in {
        "CuPySymmetryMultipoleBoundaryBuilder",
        "CuPySubgroupMultipoleBoundaryBuilder",
    }
    boundary_on_point_gpu = (
        type(native_boundary_builder).__name__ == "CuPyPointMultipoleBoundaryBuilder"
    )
    native_boundary_description = (
        "FP64 GPU multipoles on the symmetry wedge, boundary values per "
        "unique exterior point and gathered wedge RHS"
        if boundary_on_wedge_gpu
        else
        "FP64 GPU multipoles of the full grid, boundary values per unique "
        "exterior point and gathered RHS"
        if boundary_on_point_gpu
        else
        "FP64 GPU streaming multipoles and exterior-stencil RHS (bounded O(N) storage)"
        if boundary_on_gpu
        else
        "orbit-summed C++/OpenMP multipoles and direct wedge RHS"
        if native_symmetry_boundary
        else "cached C++/OpenMP multipole boundary/RHS"
        if native_boundary_builder is not None
        else "Python boundary/RHS"
    )
    native_cg_description = (
        "symmetry-wedge C++/OpenMP CG"
        if hartree_reduction is not None
        else "cached C++/OpenMP CG"
    )
    if hartree_linear_backend == "cupy":
        native_cg_description = (
            "FP64 CUDA graph CG on the normalized symmetry wedge"
            if hartree_reduction is not None
            else "FP64 CUDA graph CG on the full grid"
        )
    hartree_implementation = {
        "scipy": "fast multipole boundary plus reference-equivalent SciPy CG",
        "native": f"{native_boundary_description} plus {native_cg_description}",
        "cupy": "fast multipole boundary plus shared-device-CSR CuPy CG",
    }[selection.hartree_backend]
    finite_difference_implementation = {
        "reference": "validated vectorized Python/SciPy compressed-grid CSR builder",
        "native": "C++17 compressed-grid CSR builder (exact stencil parity)",
    }[selection.finite_difference_builder]
    if is_periodic:
        from .Hartree.periodic_device import describe_periodic_hartree

        # Where a wire, slab, Calc_Hartree_G or (since 2026-10-30 by default) plain
        # cell solve runs (the device solver of Hartree/periodic_device.py); a
        # plain cell solved on the host keeps the text below.
        hartree_implementation = describe_periodic_hartree(
            reference,
            "periodic Ewald-consistent conjugate-gradient solve "
            "(SCF.pbc.solve_periodic_hartree); no accelerated periodic "
            "Hartree backend yet",
        )
        finite_difference_implementation = (
            "validated vectorized Python/SciPy periodic-wraparound CSR "
            "builder (SCF.pbc.prepare_periodic_single_point)"
        )
    component_details = (
        (
            "cuda_initialization_overlap",
            str(preparation_overlap["cuda_initialization_overlap"]),
        ),
        (
            "backend_resolution_seconds",
            f"{preparation_overlap['backend_resolution_seconds']:.6f}",
        ),
        (
            "reference_preparation_seconds",
            f"{preparation_overlap['reference_preparation_seconds']:.6f}",
        ),
        (
            "backend_reference_overlapped_seconds",
            f"{preparation_overlap['backend_reference_overlapped_seconds']:.6f}",
        ),
        ("symmetry_mode", symmetry_mode),
        ("symmetry_detection", symmetry_detection),
        ("detected_symmetry_group_order", str(detected_group_order)),
        (
            # PARSEC_SYMMETRY_FAST_MAPS, where symmetry maps were built.
            "symmetry_fast_maps",
            "not used"
            if symmetry_mode == "off" or (is_periodic and not periodic_symmetry)
            else "on" if _fast_maps_requested() else "off",
        ),
        # PARSEC_OVERLAP_SYMMETRY_SETUP: where the symmetry maps were built,
        # and for a thread of their own its seconds and the waits at its
        # joins.
        *(
            (("symmetry_setup", "not used"),)
            if symmetry_mode == "off" or (is_periodic and not periodic_symmetry)
            else (("symmetry_setup", "inline"),)
            if symmetry_maps is None
            else symmetry_maps.details()
        ),
        (
            # The directory the caller named.  Without one, the default of
            # every entry point, no symmetry cache is read or written.
            "symmetry_cache_directory",
            "disabled"
            if symmetry_cache_directory is None
            else "not used"
            if symmetry_mode == "off" or (is_periodic and not periodic_symmetry)
            else str(Path(symmetry_cache_directory).resolve()),
        ),
        ("orbital_symmetry", orbital_symmetry),
        # What of a cluster with a point group runs on the full grid for want of the native extension (a NOTE line of
        # parsec.out, _cluster_full_grid_notes).
        *(("symmetry_full_grid_note", note) for note in full_grid_notes),
        # PARSEC_FAST_GRID, as the reference preparation read it.  The grid
        # of a periodic cell has the reference builder alone.
        (
            "grid_builder",
            "slabs" if fast_grid_requested() and not is_periodic else "reference",
        ),
        ("finite_difference_builder", finite_difference_implementation),
        (
            "ionic_setup_implementation",
            (
                ("FP64 CUDA atom-ordered local/density sums; C++ KB sampling"
                 if selection.selected == "cupy"
                 and _gpu_ionic_requested(periodic=is_periodic)
                 else "cached-grid C++/OpenMP radial interpolation and KB sampling")
                if selection.finite_difference_builder == "native"
                and "RadialGridEvaluator" in native_kernel_names
                else "vectorized NumPy radial interpolation and KB sampling"
            ),
        ),
        # Double_Grid_Order > 1 and Old_Interpolation_Format: nothing otherwise.
        *_setup_option_details(problem, selection),
        ("ionic_gpu_count_requested", ionic_gpu_count_setting()),
        (
            # The devices the GPU ionic sums of this preparation ran on.
            "ionic_gpu_devices",
            " ".join(map(str, preparation_overlap.get("ionic_sum_devices") or ()))
            or "none",
        ),
        ("ionic_projector_lookup", os.environ.get("PARSEC_NATIVE_PROJECTOR_LOOKUP", "1")),
        # A sector worker builds no ionic fields.
        *(
            ()
            if sector_worker
            else (("ionic_setup", "inline"),)
            if ionic_fields is None
            else ionic_fields.details()
        ),
        ("hartree_backend", "cupy-boundary/cupy-linear"
         if boundary_on_gpu and hartree_linear_backend == "cupy"
         else "native-boundary/cupy-linear" if hartree_linear_backend == "cupy"
         else "cupy-boundary/native-linear" if boundary_on_gpu
         else selection.hartree_backend),
        ("hartree_implementation", hartree_implementation),
        ("hartree_linear_backend", _hartree_linear_report(
            selection, periodic=is_periodic, backend=hartree_linear_backend,
            reason=hartree_linear_reason)),
        (
            "hartree_full_grid_transfer_policy",
            "device-resident" if resident_hartree is not None else "host-interface",
        ),
        (
            "hartree_resident_predictor",
            resident_hartree.predictor if resident_hartree is not None else "disabled",
        ),
        # PARSEC_CUPY_RESIDENT_HARTREE: whether the chain runs and why.
        (
            "hartree_resident_chain",
            "not used (no Hartree solver on this rank)"
            if sector_worker
            else "not used (periodic cell)"
            if is_periodic
            else f"not used (the {selection.hartree_backend} Hartree solver of the selected backend)"
            if selection.hartree_backend != "native"
            else "off (the C++/OpenMP CG of a CPU run)"
            if selection.selected != "cupy"
            else f"on ({resident_reason})"
            if resident_hartree is not None
            else f"off ({resident_reason})",
        ),
        *(
            (("hartree_resident_device_bytes", str(resident_hartree.device_bytes)),)
            if resident_hartree is not None
            else ()
        ),
        *_hartree_boundary_details(
            reference, hartree_boundary_tail, native_boundary_builder
        ),
        ("hartree_variational", hartree_variational_reason),
        (
            # The kernels that build the boundary values of every solve.
            "hartree_boundary_kernel",
            "none (no Hartree solver on this rank)"
            if sector_worker
            else "none (periodic cell: no boundary values)"
            if is_periodic
            else "wedge (GPU, symmetry wedge and unique exterior points)"
            if boundary_on_wedge_gpu
            else "points (GPU, full grid and unique exterior points)"
            if boundary_on_point_gpu
            else "full (GPU, full grid)"
            if boundary_on_gpu
            else "native wedge (C++ orbit table)"
            if native_symmetry_boundary
            else "native full grid (C++)"
            if native_boundary_builder is not None
            else "direct Coulomb sum (GPU, unique exterior points)"
            if direct_boundary is not None
            else "python",
        ),
        (
            "xc_implementation",
            getattr(
                spin_xc_evaluator if spin_polarized else native_xc_evaluator,
                "implementation_label",
                None,
            )
            or (
                # The SCF loop of two spin channels evaluates the readable
                # spin functionals on the full grid.
                _readable_spin_xc_label(xc_functional)
                if spin_polarized
                else "cached C++/OpenMP float64 CA/PZ-LDA"
                if xc_functional == "ca"
                # (not the host adapter of the orbits: a periodic cell under PARSEC_PERIODIC_CA_XC=host)
                and type(native_xc_evaluator).__name__ == "NativeCALDAEvaluator"
                else (
                    "vectorized NumPy discrete-variational PBE"
                    if xc_functional == "pbe"
                    else "vectorized NumPy CA/PZ-LDA"
                    if xc_functional == "ca"
                    else _readable_xc_label(xc_functional, spin_polarized=False)
                )
            ),
        ),
    )
    from .Laplacian import DeferredNativeNegativeLaplacian

    deferred_laplacian = reference.negative_laplacian
    if isinstance(deferred_laplacian, DeferredNativeNegativeLaplacian):
        if deferred_laplacian.matrix_origin == "built":
            materialization = "performed"
        elif deferred_laplacian.matrix_origin == "shared":
            # An earlier calculation of this resident process built it.
            materialization = "reused_from_resident_reference"
        elif (
            symmetry_eigensolver is not None
            and symmetry_eigensolver.operator_cache_info.stencil_builder
            != "cached"
        ):
            materialization = "skipped_by_direct_sector_stencil"
        else:
            materialization = "skipped_by_exact_reduced_operator_cache"
        component_details += (
            (
                "finite_difference_full_grid_materialization",
                materialization,
            ),
            (
                "finite_difference_provenance_key",
                _reported_cache_key(deferred_laplacian.hashed_cache_key),
            ),
            (
                "finite_difference_provenance_hash_seconds",
                f"{deferred_laplacian.hash_seconds:.6f}",
            ),
            (
                "finite_difference_nnz_count_seconds",
                f"{deferred_laplacian.nnz_count_seconds:.6f}",
            ),
            (
                "finite_difference_nnz_cache",
                deferred_laplacian.nnz_cache_status,
            ),
            (
                "finite_difference_nnz_cache_path",
                (
                    str(deferred_laplacian.nnz_cache_path)
                    if deferred_laplacian.nnz_cache_path is not None
                    else "disabled"
                ),
            ),
            (
                "finite_difference_materialization_seconds",
                f"{deferred_laplacian.materialization_seconds:.6f}",
            ),
            (
                "reference_static_cache",
                getattr(
                    deferred_laplacian,
                    "reference_static_cache_status",
                    "disabled",
                ),
            ),
            (
                "reference_static_cache_lookup_seconds",
                f"{getattr(deferred_laplacian, 'reference_static_cache_lookup_seconds', 0.0):.6f}",
            ),
        )
    if symmetry_geometry_cache_info is not None:
        cache_info = symmetry_geometry_cache_info
        component_details += (
            ("symmetry_geometry_cache", cache_info.status),
            ("symmetry_geometry_cache_key", _reported_cache_key(cache_info.key)),
            (
                "symmetry_geometry_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            ("symmetry_geometry_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
            ("symmetry_geometry_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
            ("symmetry_geometry_build_seconds", f"{cache_info.build_seconds:.6f}"),
            ("symmetry_geometry_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
        )
    if symmetry_representation_cache_info is not None:
        cache_info = symmetry_representation_cache_info
        component_details += (
            ("symmetry_representation_cache", cache_info.status),
            (
                "symmetry_representation_cache_key",
                _reported_cache_key(cache_info.key),
            ),
            (
                "symmetry_representation_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            ("symmetry_representation_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
            ("symmetry_representation_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
            ("symmetry_representation_build_seconds", f"{cache_info.build_seconds:.6f}"),
            ("symmetry_representation_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
        )
    if selection.selected == "cupy":
        from .Eigensolvers.orthogonalize import (
            _complete_subspace_policy,
            chebdav_block_orth_requested,
        )
        from .Eigensolvers.rayleigh_ritz import generalized_ritz_requested

        reference_input = getattr(reference, "input", None)
        scf_input = getattr(reference_input, "scf", None)
        eigensolver_input = getattr(reference_input, "eigensolver", None)
        requested_states = getattr(scf_input, "number_of_states", None)
        subspace_buffer = getattr(eigensolver_input, "subspace_buffer", None)
        if (
            orbital_decomposition is not None
            or requested_states is None
            or subspace_buffer is None
        ):
            subspace_orthogonalization = (
                "size-adaptive per representation: audited PARSEC MGS for "
                "small bases; generalized Cholesky-whitened Rayleigh--Ritz "
                "with Householder QR fallback for large bases"
            )
        else:
            working_states = min(
                reference.grid.size,
                int(requested_states)
                + int(subspace_buffer),
            )
            selected_orthogonalization = _complete_subspace_policy(
                reference.grid.size,
                working_states,
            )
            if generalized_ritz_requested(
                reference.grid.size, working_states
            ):
                subspace_orthogonalization = (
                    "audited generalized Cholesky-whitened Rayleigh--Ritz "
                    f"selected for {reference.grid.size}x{working_states} "
                    "filtered basis; Householder QR stability fallback"
                )
            else:
                subspace_orthogonalization = (
                    f"{selected_orthogonalization} selected for "
                    f"{reference.grid.size}x{working_states} saved basis"
                )
        component_details += (
            (
                "gpu_initial_random_basis",
                (
                    "bit-exact device DLARNV for initial CHEBFF basis"
                    if os.environ.get("PARSEC_CUPY_DEVICE_RANDOM", "0").lower()
                    in {"1", "true", "on"}
                    else "bit-exact host DLARNV before device upload"
                ),
            ),
            (
                "gpu_chebdav_ritz_scalar_source",
                (
                    "reuse values already returned by host LAPACK"
                    if os.environ.get(
                        "PARSEC_CUPY_REUSE_HOST_RITZ_VALUES", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "explicit device scalar transfers"
                ),
            ),
            (
                "gpu_chebdav_appended_orthogonalization",
                (
                    "not used by the selected first eigensolver"
                    if eigensolver_input is None
                    or eigensolver_input.method != "chebdav"
                    else (
                        "audited FP64 block-CGS2/device-MGS2 with "
                        "Householder and PARSEC-MGS fallbacks"
                        if chebdav_block_orth_requested(
                            (
                                max(orbital_decomposition.sector_sizes)
                                if orbital_decomposition is not None
                                else reference.grid.size
                            ),
                            eigensolver_input.matvec_block_size,
                        )
                        else "PARSEC selective MGS"
                    )
                ),
            ),
            (
                "gpu_chebdav_prefix_projection",
                (
                    (
                        "full C-order coefficient GEMM plus fused "
                        "active-prefix CUDA update for blocks up to six"
                        if os.environ.get(
                            "PARSEC_CUPY_CHEBDAV_FUSED_PREFIX_UPDATE", "1"
                        ).strip().lower()
                        not in {"0", "false", "no", "off"}
                        else "full C-order workspace GEMM with zero "
                        "inactive coefficients"
                    )
                    if os.environ.get(
                        "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_CGS", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "active noncontiguous prefix GEMM"
                ),
            ),
            (
                "gpu_chebdav_ritz_projection",
                (
                    "full contiguous C-order Davidson workspace GEMM; "
                    "use only the active row interval"
                    if os.environ.get(
                        "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_RITZ", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "active noncontiguous Davidson basis GEMM"
                ),
            ),
            (
                "gpu_subspace_orthogonalization",
                subspace_orthogonalization,
            ),
        )
    if orbital_decomposition is not None:
        from .Symmetry import operator_build_workers

        orbital_reduction = orbital_decomposition.reduction
        component_details += (
            ("symmetry_full_grid_points", str(orbital_reduction.full_size)),
            ("symmetry_wedge_points", str(orbital_reduction.wedge_size)),
            (
                "symmetry_reduction_ratio",
                f"{orbital_reduction.reduction_ratio:.6g}",
            ),
        )
    if orbital_decomposition is not None:
        cache_info = symmetry_eigensolver.operator_cache_info
        if symmetry_eigensolver.scheduler_mode == "sequential":
            sector_scheduler = "sequential on one CUDA device"
        elif symmetry_eigensolver.scheduler_mode == "multi-gpu":
            device_list = ",".join(map(str, symmetry_eigensolver.device_ids))
            sector_scheduler = (
                "independent representations distributed across "
                f"CUDA devices {device_list}"
            )
        else:
            sector_scheduler = (
                "concurrent nonblocking CUDA streams "
                f"({symmetry_eigensolver.scheduler_workers} workers)"
            )
        component_details += (
            (
                "orbital_symmetry_representations",
                str(orbital_decomposition.representation_count),
            ),
            (
                "orbital_sector_dimensions",
                " ".join(map(str, orbital_decomposition.sector_sizes)),
            ),
            *(
                (("orbital_partner_rows", symmetry_eigensolver.partner_rows_record),)
                if getattr(symmetry_eigensolver, "partner_rows_record", None)
                else ()
            ),
            (
                "orbital_sector_stabilizer_handling",
                "exact character selection on every orbit stabilizer",
            ),
            (
                "orbital_sector_state_policy",
                "floor(global_states/representations) + Subspace_Buffer_Size, "
                "then grow sectors to bracket the global cutoff",
            ),
            ("orbital_sector_scheduler", sector_scheduler),
            (
                "orbital_sector_cuda_devices",
                " ".join(map(str, symmetry_eigensolver.device_ids)),
            ),
            # The devices of every sector of one process with several GPUs (PARSEC_CUPY_SECTOR_PLACEMENT); an
            # MPI rank's placement is recorded by the runner.
            *(
                (
                    (
                        "orbital_sector_placement",
                        process_placement_note
                        or getattr(symmetry_eigensolver, "placement_record", None)
                        or "not recorded",
                    ),
                )
                if mpi_context is None and len(symmetry_eigensolver.device_ids) > 1
                else ()
            ),
            (
                "orbital_sector_lanczos_scheduler",
                (
                    "serial bound preparation on owning streams before sector workers"
                    if symmetry_eigensolver.bound_schedule == "before_sectors"
                    else "concurrent nonblocking streams; filters remain sequential"
                    if symmetry_eigensolver.collective_lanczos
                    else "same scheduler as sector solves"
                ),
            ),
            (
                "orbital_sector_finite_difference_storage",
                symmetry_eigensolver.finite_difference_storage,
            ),
            (
                # one-byte or two-byte coefficient codes (PARSEC_SECTOR_PALETTE_CODES)
                "orbital_sector_coefficient_codes",
                # (a stand-in solver without stencils: none)
                getattr(symmetry_eigensolver, "coefficient_codes", "none"),
            ),
            (
                "orbital_sector_tile_pack_workers",
                _reported_pack_workers(
                    symmetry_eigensolver.finite_difference_storage
                ),
            ),
            (
                "orbital_sector_neighbor_storage",
                symmetry_eigensolver.finite_difference_neighbors,
            ),
            (
                # PARSEC_CUPY_STENCIL_KERNEL as the stencils of the sectors
                # were built with it, an unset variable included: the slot
                # count of the fast path of the pipelined kernel, or why they
                # hold the former kernels.
                "orbital_sector_stencil_kernel",
                symmetry_eigensolver.stencil_kernel,
            ),
            (
                "orbital_sector_nonlocal_application",
                (
                    (
                        "canonical-order custom CUDA B.T projection plus KB "
                        "scatter fused into CUDA stencil"
                        if symmetry_eigensolver.custom_projector_projection
                        else "cuSPARSE B.T projection plus KB scatter fused "
                        "into CUDA stencil"
                    )
                    if symmetry_eigensolver.fused_projector_scatter
                    else "two sparse KB contractions"
                ),
            ),
            (
                "orbital_sector_projector_reduction",
                symmetry_eigensolver.projector_reduction_modes,
            ),
            (
                "orbital_sector_later_filter_precision",
                symmetry_eigensolver.later_filter_precision,
            ),
            (
                # PARSEC_CUPY_RITZ_HANDOVER, an unset variable included: what
                # the Ritz steps hand over.  A result says what they did
                # (SCF/single_point.py).
                "orbital_sector_ritz_handover",
                getattr(symmetry_eigensolver, "ritz_handover", "rotation"),
            ),
            (
                # PARSEC_CUPY_FILTER_GRAPH_REUSE, for a filter that records
                # graphs: none has run yet.  A result says what the filters
                # of its sectors recorded (SCF/single_point.py).
                "orbital_sector_filter_graphs",
                graph_route_name(shared_filter_graphs),
            ),
            (
                # PARSEC_CUPY_RITZ_PIPELINE and PARSEC_CUPY_EXCHANGE_STRIDED
                # as the Ritz steps of a shared sector basis will read them,
                # unset variables included: none has run yet.  A result says
                # what the steps of its shared sectors ran
                # (SCF/single_point.py).
                "orbital_sector_ritz_pipeline",
                shared_ritz_pipeline,
            ),
            ("orbital_sector_exchange_copies", shared_exchange_copies),
            (
                "orbital_sector_local_potential_storage",
                symmetry_eigensolver.local_potential_storage,
            ),
            (
                "orbital_density_storage",
                "representation-sector Ritz views; density accumulated "
                "directly on physical scalar orbits without packed "
                "wedge-by-state duplication",
            ),
            ("orbital_operator_cache", cache_info.status),
            ("orbital_operator_cache_key", _reported_cache_key(cache_info.key)),
            (
                "orbital_operator_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            (
                "orbital_operator_hash_seconds",
                f"{cache_info.hash_seconds:.6f}",
            ),
            (
                "orbital_operator_cache_load_seconds",
                f"{cache_info.load_seconds:.6f}",
            ),
            (
                "orbital_operator_build_seconds",
                f"{cache_info.build_seconds:.6f}",
            ),
            ("orbital_operator_stencil_builder", cache_info.stencil_builder),
            (
                "orbital_operator_build_workers",
                str(
                    operator_build_workers(
                        symmetry_eigensolver.assembled_sector_count
                    )
                    # (A rank of a hybrid's MPI run may assemble no sector.)
                    if symmetry_eigensolver.assembled_sector_count
                    else 0
                ),
            ),
            (
                "orbital_operator_cache_write_seconds",
                f"{cache_info.write_seconds:.6f}",
            ),
            (
                "scf_scalar_field_storage",
                "one physical value per symmetry orbit with multiplicity weights",
            ),
            (
                # PARSEC_SECTOR_FIELD_GROUP and PARSEC_SECTOR_INPUT_GROUP_AVERAGE: the orbits the fields keep.
                "scf_scalar_field_orbits",
                (
                    f"full group ({field_reduction.group_order} operations): {field_reduction.wedge_size} "
                    f"values, {orbital_reduction.wedge_size} on the orbits of the sectors' group "
                    f"(PARSEC_SECTOR_FIELD_GROUP={sector_field_group})"
                )
                if field_reduction is not None
                else (
                    f"sectors' group ({getattr(orbital_reduction, 'group_order', '?')} operations): "
                    f"{getattr(orbital_reduction, 'wedge_size', '?')} values"
                    + (
                        f"; full-grid inputs averaged over the {len(orbital_reduction.full_operations)} operations "
                        "of the full group (PARSEC_SECTOR_INPUT_GROUP_AVERAGE=auto)"
                        if getattr(scf_reducer, "group_average_inputs", False)
                        else ""
                    )
                ),
            ),
            (
                # PARSEC_SYMMETRY_SCF_BUFFERS as it is set now; the wedge sums
                # and the mixer read it again at every call.
                "scf_scalar_field_buffers",
                "on" if scf_buffers_requested() else "off",
            ),
        )
    if sector_worker:
        # The component entries above name the selected implementations;
        # this rank built only the ones its sectors use.
        component_details += (
            (
                "mpi_rank_preparation",
                "symmetry-sector operators only; ionic fields, densities, "
                "Hartree and XC are prepared on the root rank",
            ),
        )
    elif selection.hartree_backend == "native":
        if hartree_linear_backend == "cupy":
            gpu_poisson = (poisson_solver.solver if hartree_reduction is not None
                           else poisson_solver).backend
            component_details += (
                ("hartree_cg_device", str(gpu_poisson.device_id)),
                ("hartree_cg_graph_steps", str(gpu_poisson.graph_iterations)),
                ("hartree_cg_device_bytes", str(gpu_poisson.device_bytes)),
                ("hartree_cg_stencil_packing", gpu_poisson.stencil_packing),
            )
        component_details += (
            ("hartree_boundary_device_bytes", str(
                native_boundary_builder.device_storage_bytes if boundary_on_gpu else 0)),
            ("hartree_boundary_device", str(
                native_boundary_builder.device_id) if boundary_on_gpu else "host"),
            ("hartree_boundary_setup", boundary_setup_status),
            (
                "hartree_boundary_setup_seconds",
                f"{boundary_setup_seconds:.6f}",
            ),
            (
                "hartree_boundary_setup_wait_seconds",
                f"{boundary_setup_wait_seconds:.6f}",
            ),
            (
                "hartree_boundary_setup_overlapped_seconds",
                f"{boundary_setup_overlapped_seconds:.6f}",
            ),
            ("hartree_cg_storage", poisson_solver.storage_mode),
            ("hartree_cg_openmp_workers", str(poisson_solver.worker_count)),
            (
                "hartree_initial_guess",
                (
                    "two-step chronological RHS predictor with "
                    "previous-potential fallback"
                    if os.environ.get(
                        "PARSEC_HARTREE_CHRONOLOGICAL_GUESS", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "previous SCF Hartree potential"
                ),
            ),
            (
                "hartree_cg_coefficient_palette_size",
                str(poisson_solver.coefficient_palette_size),
            ),
        )
        if hartree_reduction is None:
            component_details += (
                (
                    "hartree_symmetry",
                    "full grid (spin-polarized: the density need not keep "
                    "the symmetry of the structure)"
                    if hartree_full_grid_for_spin
                    else "full grid",
                ),
            )
        else:
            component_details += (
                (
                    "hartree_symmetry",
                    "normalized Cartesian axis-reflection wedge",
                ),
                (
                    "hartree_symmetry_group_order",
                    str(hartree_reduction.group_order),
                ),
                ("hartree_full_grid_points", str(hartree_reduction.full_size)),
                ("hartree_wedge_points", str(hartree_reduction.wedge_size)),
                (
                    "hartree_reduction_ratio",
                    f"{hartree_reduction.reduction_ratio:.6g}",
                ),
            )
        if native_boundary_cache_info is not None:
            cache_info = native_boundary_cache_info
            component_details += (
                ("hartree_geometry_cache", cache_info.status),
                (
                    "hartree_geometry_cache_key",
                    _reported_cache_key(cache_info.key),
                ),
                (
                    "hartree_geometry_cache_path",
                    str(cache_info.path)
                    if cache_info.path is not None
                    else "disabled",
                ),
                ("hartree_geometry_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
                ("hartree_geometry_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
                ("hartree_geometry_build_seconds", f"{cache_info.build_seconds:.6f}"),
                ("hartree_geometry_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
            )
    if selection.selected != "native" and (
        selection.finite_difference_builder == "native"
        or selection.hartree_backend == "native"
    ):
        # A hybrid is reported as a CuPy execution backend, so copy the native
        # runtime configuration into its provenance explicitly.
        from .backends.native import native_build_info

        native_build = native_build_info()
        component_details += (
            (
                "native_openmp_detected_processors",
                str(native_build.get("openmp_detected_processors", 1)),
            ),
            (
                "native_openmp_reserved_threads",
                str(native_build.get("openmp_reserved_threads", 0)),
            ),
            (
                "native_openmp_max_threads",
                str(native_build.get("openmp_max_threads", 1)),
            ),
            (
                "native_openmp_thread_source",
                str(native_build.get("openmp_thread_source", "unknown")),
            ),
        )
    # The whole preparation last: what follows only composes the system.
    # The timeline is carried by the prepared system and is no entry of the
    # backend report: the result archive stores every entry of the report at
    # the width of the longest, and one line of all stages is several times
    # as long as any other entry.
    stages.add("preparation", stages.origin)
    implementation.info = replace(
        implementation.info,
        details=implementation.info.details + component_details,
    )
    # ------------------------------------------------------------------
    # Ionic steps.  The execution objects built above are shared by every
    # structure of a relaxation: what they hold for the atom positions is
    # replaced in place, and the Hartree and XC closures read the rebound
    # ``reference`` and tail from here on.
    # ------------------------------------------------------------------

    def tail_maximum():
        # The builder of the present structure: a resident chain that gave
        # way rebuilds it on the host.
        return (
            hartree_boundary_tail.maximum
            if hartree_boundary_tail is not None
            else getattr(native_boundary_builder, "atomic_tail_maximum", None)
            if type(native_boundary_builder).__name__ in {
                "CuPySymmetryMultipoleBoundaryBuilder",
                "CuPyPointMultipoleBoundaryBuilder",
                "CuPySubgroupMultipoleBoundaryBuilder",
            }
            else None
        )

    def projector_gradients(atom_indices=None):
        """Gradients of the projectors of the present atoms, column for column.

        ``atom_indices`` (the representative atoms of the orbits of the
        symmetry sectors, :func:`.Symmetry.axis_reflection.sector_gradient_atoms`)
        samples those atoms only: the columns of the projectors of these
        atoms, in the order of the projectors and with their atom indices
        (on either set of routines of a cell).
        """

        from parsec_python.Forces import (
            NonlocalProjectorGradients,
            gradient_columns_for,
        )
        # The readable builder, or the double-grid one for Double_Grid_Order > 1.
        from .V_ion.double_grid import build_projector_gradients

        operator = reference.nonlocal_operator
        selected = None
        labels = tuple(operator.labels)
        if atom_indices is not None:
            selected = np.unique(np.asarray(atom_indices, dtype=np.int64).reshape(-1))
            wanted = np.zeros(len(reference.atoms), dtype=bool)
            wanted[selected] = True
            # the projector columns of the selected atoms, in their order
            labels = tuple(label for label in labels if wanted[int(label[0])])

        def _matching_columns(gradients, labels):
            columns = gradient_columns_for(labels, gradients.labels)
            if isinstance(columns, slice):
                return gradients
            return NonlocalProjectorGradients(
                components=tuple(
                    component[:, columns] for component in gradients.components
                ),
                signs=gradients.signs[columns],
                labels=tuple(labels),
            )

        if is_periodic:
            if uses_fortran_routines(reference.grid):
                # The Fortran-derived routines: the gradients of the projectors with their lattice images, of
                # every atom or of the representatives of the orbits of the sectors.
                from .Forces.periodic_device import periodic_projector_gradients

                return _matching_columns(
                    periodic_projector_gradients(reference, atom_indices=selected), labels
                )
            from parsec_python.Forces.nonlocal_force import (
                build_periodic_projector_gradients,
            )

            # images and the folded atoms of the projector builder
            gradients = build_periodic_projector_gradients(
                reference.grid,
                reference.cell_atoms,
                reference.pseudopotentials,
                reference.input.pseudopotentials,
                reference.input.periodic_cell.lattice_vectors,
                atom_indices=selected,
            )
        else:
            gradients = build_projector_gradients(
                reference.grid,
                reference.atoms,
                reference.pseudopotentials,
                reference.input.pseudopotentials,
                atom_indices=selected,
            )
        return _matching_columns(gradients, labels)

    def move_atoms(positions):
        """Rebuild in this process what depends on the atom positions.

        ``parsec.F90`` after ``domove``: ``isolat``, ``nonloc``, ``corecd``,
        ``ionpot`` and ``forceion`` on the unchanged grid.  A rank that only
        serves symmetry sectors rebuilds its projectors alone.

        A periodic cell (``back_to_cell``, ``nonloc``, ``corecd``, ``ionpbc``,
        ``ewald_sum``) is rebuilt by :func:`parsec_python.SCF.pbc.update_periodic_geometry`;
        one solved in the sectors of its point group moves where the atoms
        keep that group (:func:`.Forces.sector_symmetry.check_sector_atoms`,
        which a relaxation with forces and positions averaged over it does),
        on Andrew Grieder's routines and, under
        ``PARSEC_FORTRAN_SECTOR_FORCES=auto``, on the Fortran-derived ones
        (their fields rebuilt on the full grid as without sectors).  Every
        route then installs the moved
        projectors in the full-grid operator and in the sector operators
        (compressed to one atom per orbit as at preparation), rebinds
        ``reference``, which every closure of the SCF loop reads (the local
        ionic potential, the core density of the XC evaluators, the Hartree
        solve of a cell, the Ewald energy), and starts the projector
        gradients of the next forces.
        """

        nonlocal reference, hartree_boundary_tail, native_xc_evaluator
        from parsec_python.SCF import update_geometry as update_reference_geometry

        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(reference.atoms), 3):
            raise ValueError("one position per atom of the prepared system is required")
        # every other field of an atom is kept (the initial moment of a non-collinear run)
        atoms = tuple(
            replace(atom, position=position)
            for atom, position in zip(reference.atoms, positions)
        )
        in_sectors = symmetry_reduction is not None and symmetry_reduction.group_order > 1
        if is_periodic:
            fortran_routines = uses_fortran_routines(reference.grid)
            if in_sectors and fortran_routines:
                from .Forces.sector_symmetry import fortran_sector_forces_setting

                if fortran_sector_forces_setting() == "off":
                    # Their forces are refused in the sectors (Forces/compute.py).
                    raise NotImplementedError(
                        "the atoms of a periodic cell of the Fortran-derived routines solved in the sectors of its "
                        "point group do not move under PARSEC_FORTRAN_SECTOR_FORCES=off: prepare it with "
                        "symmetry='off' or PARSEC_PERIODIC_SYMMETRY=off"
                    )
            if in_sectors:
                # The sectors, their weights and the projectors of one atom
                # per orbit hold for the group of the first structure only.
                from .Forces.sector_symmetry import check_sector_atoms

                check_sector_atoms(
                    symmetry_reduction,
                    atoms,
                    reference.grid,
                    reference.input.periodic_cell.lattice_vectors,
                )
            if fortran_routines:
                # parsec.F90 after domove for a bulk cell (back_to_cell, nonloc,
                # corecd, ionpbc, ewald_sum): the device fields inside
                # device_periodic_forces.
                from .Forces.periodic_device import update_periodic_reference

                moved = update_periodic_reference(reference, atoms)
            else:
                # Andrew Grieder's routines: the ionic potential, the projectors,
                # the densities and the Ewald terms of the new positions; the
                # projectors alone on a rank that shares the basis of a sector
                # (its reference holds no ionic potential).
                from parsec_python.SCF import update_periodic_geometry

                moved = update_periodic_geometry(reference, atoms)
        else:
            if symmetry_reduction is not None:
                # Sectors, wedge fields and wedge sums hold for this group only.
                from .Forces.local_device import sector_atom_images

                sector_atom_images(symmetry_reduction, atoms)
            moved_ionic_report: dict = {}
            moved = update_reference_geometry(
                reference,
                atoms,
                **_geometry_builder_hooks(
                    selection,
                    ionic_sector_rank=mpi_context is not None,
                    # GPU sums on the orbits of the full group of the first structure, else of its sectors'
                    # group, where the moved atoms are their images (PARSEC_IONIC_MOVED_SUMS).
                    orbits=(reference.grid, atoms, _ionic_sum_candidates(symmetry_reduction)),
                    report=moved_ionic_report,
                ),
                build_atomic_tail=False,
            )
            # (where the sums of the latest moved structure ran, for its report and the tests)
            implementation.moved_ionic_sum_grid = moved_ionic_report.get("ionic_sum_grid")
        moved_nonlocal = (
            moved.nonlocal_operator
            if hubbard_device is None
            else hubbard_device.move(moved)
        )
        implementation.replace_nonlocal_operator(moved_nonlocal)
        if symmetry_eigensolver is not None:
            symmetry_eigensolver.replace_nonlocal_operator(moved_nonlocal)
        if not sector_worker and not is_periodic:
            tail_on_gpu = (
                selection.selected == "cupy"
                and selection.hartree_backend in {"native", "cupy"}
                and not tail_on_host
            )
            hartree_boundary_tail = _refresh_boundary_tail(
                native_boundary_builder,
                moved,
                on_device=tail_on_gpu,
                device_id=(
                    hartree_device
                    if selection.hartree_backend == "native"
                    else None
                ),
            )
            if type(native_xc_evaluator).__name__ == "NativeCALDAEvaluator":
                # The C++ evaluator owns a copy of the frozen core density.
                native_xc_evaluator = type(native_xc_evaluator)(
                    moved.core_density,
                    moved.grid.volume_element,
                    scf_reducer,
                )
        reference = moved
        if symmetry_eigensolver is not None:
            # The forces of these atoms follow their SCF loop: the host part
            # of the nonlocal term is built beside it.
            symmetry_eigensolver.prefetch_force_gradients()

    def update_system(system, atoms):
        """Move the atoms on every rank; return the system of the new structure."""

        atoms = tuple(atoms)
        if len(atoms) != len(reference.atoms) or any(
            new.symbol != old.symbol for new, old in zip(atoms, reference.atoms)
        ):
            raise ValueError("moved atoms must keep their number, order and species")
        positions = np.array(
            [np.asarray(atom.position, dtype=np.float64) for atom in atoms]
        )
        if mpi_context is not None:
            mpi_context.execute(symmetry_eigensolver, "geometry", positions)
        else:
            move_atoms(positions)
        return replace(
            system,
            reference=reference,
            xc_evaluator=native_xc_evaluator,
            boundary_tail_maximum=tail_maximum(),
        )

    if symmetry_eigensolver is not None:
        symmetry_eigensolver.projector_gradient_provider = projector_gradients
        symmetry_eigensolver.geometry_handler = move_atoms
    # LDA+U in the device Hamiltonians and the TS sums on a device.
    hubbard_device = device_hubbard_for(
        reference, implementation, symmetry_eigensolver, selection.selected
    )
    dispersion_device = None
    if selection.selected == "cupy" and getattr(
        getattr(problem, "scf", None), "van_der_waals", False
    ):
        from .Energy import device_dispersion_correction

        dispersion_device = device_dispersion_correction
    # A hybrid functional: its exact exchange in the device Hamiltonians.
    from .SCF.hybrid import device_exchange_for

    exchange_device = device_exchange_for(
        reference,
        implementation,
        symmetry_eigensolver,
        selected_backend=selection.selected,
        # The structure of the present atoms (move_atoms rebinds it).
        start_xc=_start_xc_with_reducer(
            hybrid_start_xc, lambda: reference, scf_reducer, spin_sectors=spin_sectors
        ),
    )
    if exchange_device is not None and mpi_context is not None:
        # Several ranks: every piece of the Fock loop is a command to all of
        # them, and this rank's service answers it (SCF/hybrid_mpi.py).
        from .SCF.hybrid_mpi import mpi_device_exchange_for

        exchange_device = mpi_device_exchange_for(
            exchange_device, mpi_context, symmetry_eigensolver, reference
        )
    if exchange_device is not None and symmetry_eigensolver is not None:
        # The memory rules of the sectors count what the hybrid keeps on the
        # devices: its device XC evaluators and the exchange (xi, the
        # expansions, the Kronecker plans and the tile work), so that the
        # sector states are spilled where they and the exchange do not fit.
        _count_hybrid_device_bytes(
            symmetry_eigensolver,
            exchange_device,
            reference,
            spin_xc_evaluator if spin_sectors else native_xc_evaluator,
            hybrid_start_xc,
        )
    # A meta-GGA: its orbital term in the device Hamiltonians, tau where the
    # orbitals are; the CPU backends add the host term (SCF/single_point.py).
    meta_gga_device = None
    if selection.selected == "cupy" and getattr(
        getattr(getattr(reference, "input", None), "scf", None), "is_meta_gga", False
    ):
        from .Hamiltonian.meta_gga_device import DeviceMetaGGA

        stage_started = perf_counter()
        meta_gga_device = DeviceMetaGGA(
            backend=implementation,
            sectors=symmetry_eigensolver,
            reducer=scf_reducer,
            grid=reference.grid,
        )
        meta_gga_device.prepare()
        from parsec_python.V_xc.meta_gga import meta_gga_gradient_form

        orbital_gradient = meta_gga_gradient_form(reference.input.scf)[0] == "orbital"
        resident = dict(meta_gga_device.device_bytes)
        if orbital_gradient:
            for device_id, count in meta_gga_device.flux_device_bytes.items():
                resident[device_id] = resident.get(device_id, 0) + count
        if symmetry_eigensolver is not None:
            # The memory rules of the sectors count the derivative matrices
            # (and W with the shared-row maps of the orbital form).
            counts = dict(symmetry_eigensolver.resident_device_bytes)
            for device_id, count in resident.items():
                counts[device_id] = counts.get(device_id, 0) + count
            symmetry_eigensolver.resident_device_bytes = counts
        implementation.info = replace(
            implementation.info,
            details=implementation.info.details
            + (
                (
                    "meta_gga_term",
                    "-div(vtau grad) as sum_a B_a^T diag(vtau) B_a of the first-derivative stencil"
                    + (" with 2 sum_a [W_a B_a + B_a^T W_a] of the orbital density gradient" if orbital_gradient else "")
                    + f" ({'sector-reduced' if symmetry_eigensolver is not None else 'full grid'}, CSR on the devices, "
                    f"{sum(resident.values())} B)",
                ),
            ),
        )
        stages.add("meta_gga_term", stage_started)
    # Spin_Symmetry_Sectors: the adapter of the two channels' orbit fields,
    # with the symmetry guard, and how the run goes back to the full grid.
    spin_adapter = None
    spin_sector_fallback = None
    if spin_sectors:
        from .SCF.spin_sectors import SpinSectorReducer

        spin_adapter = SpinSectorReducer(
            scf_reducer,
            hybrid=bool(getattr(reference.input.scf, "is_hybrid", False)),
            fixed_occupations=getattr(reference.input.scf, "fixed_occupations", None) is not None,
            # The grid and atoms of the present structure (move_atoms rebinds
            # ``reference``): the point group of the density test.  With the
            # pseudo-atoms of a field axis or of point charges after them
            # (symmetry_constraints, PARSEC_POINT_CHARGE_SYMMETRY=auto): the
            # group of the Hamiltonian, which the densities keep.
            structure=(
                (lambda: (reference.grid, reference.atoms))
                if not detection_extra
                else (lambda: (reference.grid, tuple(reference.atoms) + detection_extra))
            ),
            # the one sector of the identity of a shared fallback holds every state of the full grid
            guard=not spin_identity_sector,
        )
        spin_sector_fallback = None if spin_identity_sector else _SpinSectorFallback(
            given_problem,
            tuple(reference.atoms),
            backend=backend,
            symmetry=symmetry,
            symmetry_cache_directory=symmetry_cache_directory,
            mpi_context=mpi_context,
            plan=spin_fallback,
            symmetry_fixed_atoms=symmetry_fixed_atoms,
            symmetry_constraints=tuple(symmetry_constraints),
            sector_forces=sector_forces,
        )
    elif (
        getattr(getattr(getattr(reference, "input", None), "scf", None), "fixed_occupations", None) is not None
        or getattr(getattr(getattr(reference, "input", None), "scf", None), "is_hybrid", False)
    ) and _weighted_sectors(symmetry_eigensolver):
        # occup.in in the weighted sectors: a run whose file fills partners unequally starts again in the
        # abelian sectors (PARSEC_ORBITAL_GROUP_OCCUPATIONS), on several MPI ranks too (every rank prepares them again,
        # PARSEC_ORBITAL_GROUP_MPI=auto).  An unpolarized hybrid in them (PARSEC_ORBITAL_GROUP_HYBRID) whose start or
        # inner loop ends with a partly filled degenerate level starts again there too (SCF/hybrid.py,
        # WeightedHybridAbandoned).
        spin_sector_fallback = _SpinSectorFallback(
            given_problem,
            tuple(reference.atoms),
            backend=backend,
            symmetry=symmetry,
            symmetry_cache_directory=symmetry_cache_directory,
            mpi_context=mpi_context,
            symmetry_fixed_atoms=symmetry_fixed_atoms,
            symmetry_constraints=tuple(symmetry_constraints),
            sector_forces=sector_forces,
            spin=False,
        )
    # PARSEC_SPIN_FLIP_SYMMETRY=auto: the down channel as the image of the up
    # channel under an operation that exchanges the partner labels.
    spin_flip = None
    if spin_polarized and spin_flip_mode == "auto" and not sector_worker:
        stage_started = perf_counter()
        spin_flip, spin_flip_reason = _spin_flip_route(
            given_problem,
            reference,
            reducer=scf_reducer if spin_sectors else None,
            decomposition=orbital_decomposition if spin_sectors else None,
        )
        if spin_flip is not None:
            spin_flip = replace(spin_flip, stability=spin_flip_stability)
        stages.add("spin_flip_detection", stage_started)
        implementation.info = replace(
            implementation.info,
            details=implementation.info.details
            + (
                (
                    "spin_flip_symmetry",
                    (
                        f"auto: the up channel is solved, the down channel is its image under "
                        f"{spin_flip.description} (labels "
                        + ", ".join(
                            f"{label}->{partner}"
                            for label, partner in spin_flip.partners.items()
                            if label != partner
                        )
                        + "); the state keeps this magnetic symmetry unless a loop leaves the route (a fractional "
                        "occupation of the up channel"
                        + (
                            ", a state unstable against perturbations that break the operation "
                            "(PARSEC_SPIN_FLIP_STABILITY=auto),"
                            if spin_flip_stability == "auto"
                            else ""
                        )
                        + " or a failed test of the spin sectors: a NOTE, then both channels)"
                    )
                    if spin_flip is not None
                    else f"auto: both channels solved ({spin_flip_reason})",
                ),
            ),
        )
    if spin_polarized and (
        spin_sectors or problem.scf.is_hybrid or problem.scf.spin_symmetry_sectors is not None
    ):
        # (A semilocal run at the default that stays on the full grid, without
        # a point group or on a CPU backend, keeps the details it had.)
        implementation.info = replace(
            implementation.info,
            details=implementation.info.details
            + (
                (
                    "spin_symmetry_sectors",
                    (
                        "both channels in one sector of the identity, the full grid, its basis shared by "
                        + ("the MPI ranks" if mpi_context is not None else "the GPUs of the process")
                        + " (PARSEC_SPIN_FALLBACK_SHARED)"
                    )
                    if spin_identity_sector
                    else (
                        f"{problem.scf.spin_symmetry_sector_mode}: both channels in the "
                        f"{orbital_decomposition.representation_count} sectors; the full grid where the "
                        "start has a partly filled multiplet, a channel density breaks the point group "
                        "of the atoms or an occupation ends fractional"
                    )
                    if spin_sectors
                    else f"{problem.scf.spin_symmetry_sector_mode}: full grid",
                ),
            ),
        )
    if hybrid_full_grid_forces and (symmetry_reduction is None or int(symmetry_reduction.group_order) <= 1):
        # PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto: a Gamma hybrid below the floor of its sectors on the full grid,
        # whose forces and moved positions the command line and Forces/compute.py average over the operations of
        # the cell (_gamma_hybrid_full_grid_forces); moved structures keep this system and its details.
        from .SCF.gamma_hybrid_route import FULL_GRID_FORCES_DETAIL, FULL_GRID_FORCES_VALUE

        implementation.info = replace(
            implementation.info,
            details=implementation.info.details
            + (
                (
                    FULL_GRID_FORCES_DETAIL,
                    f"{FULL_GRID_FORCES_VALUE}; forces and moved positions averaged over the operations of the cell",
                ),
            ),
        )

    return AcceleratedPreparedSinglePointSystem(
        reference=reference,
        backend=implementation,
        backend_info=implementation.info,
        preparation_stages=stages.records(),
        eigenproblem_solver=getattr(
            implementation, "eigenproblem_solver", None
        ),
        hartree_solver=accelerated_hartree,
        hartree_variational=hartree_variational,
        boundary_tail_maximum=tail_maximum(),
        # A periodic cell moves its atoms on the Gamma-point path of
        # Forces/periodic_device.py (Ewald terms, projectors with images) with
        # the Fortran-derived routines, else with Andrew Grieder's
        # update_periodic_geometry (move_atoms).
        geometry_updater=update_system,
        projector_gradients=projector_gradients,
        # The group move_atoms checks the moved atoms against (a cluster keeps
        # it where its orbitals take the full grid): read before the SCF by
        # Forces/sector_symmetry.sector_force_tolerance_refusal.
        move_symmetry_reduction=symmetry_reduction,
        orbital_density_builder=getattr(
            implementation, "orbital_density_builder", None
        ),
        xc_evaluator=native_xc_evaluator,
        spin_xc_evaluator=spin_xc_evaluator,
        **(
            dict(
                mixer_factory=(None if scf_reducer is None else scf_reducer.mixer),
                residual_metrics_evaluator=(
                    None
                    if scf_reducer is None
                    else scf_reducer.potential_residual_metrics
                ),
                total_energy_evaluator=(
                    None if scf_reducer is None else scf_reducer.total_energy
                ),
                scalar_field_adapter=scf_reducer,
            )
            if spin_adapter is None
            # The two channels on the orbits (run_spin_scf takes these hooks).
            else spin_adapter.spin_hooks()
        ),
        hubbard_device=hubbard_device,
        dispersion_device=dispersion_device,
        exchange_device=exchange_device,
        spin_sector_fallback=spin_sector_fallback,
        spin_floor_full_grid=bool(spin_polarized and below_floor is not None),
        spin_flip=spin_flip,
        meta_gga_device=meta_gga_device,
        **_fdet_kinetic_operators(reference, selection.selected, hartree_device),
    )


def _fdet_kinetic_operators(reference, backend: str, device_id) -> dict:
    """``kinetic_operators`` of an FDET system on CuPy (the device kinetic terms), else nothing."""

    if getattr(reference, "embedding", None) is None:
        return {}
    from .Embedding.device_kinetic import device_kinetic_operators

    operators = device_kinetic_operators(reference, backend, device_id)
    return {} if operators is None else {"kinetic_operators": operators}


def _refuse_corrections(problem, *, is_periodic: bool, mpi_context) -> None:
    """Refuse the LDA+U and TS cases the accelerated driver does not cover.

    It covers both for clusters, unpolarized and spin-polarized, and for the
    Gamma point of a bulk cell, in one process (Hamiltonian.hubbard_device,
    Energy.dispersion_device); the accelerated command line sends the others
    to the reference solver, whose k-point drivers run them on its device
    workers.
    """

    species = getattr(problem, "pseudopotentials", {}).values()
    hubbard = [item for item in species if getattr(item, "has_hubbard", False)]
    if hubbard and any(float(item.hubbard_j) != 0.0 for item in hubbard):
        raise NotImplementedError(
            "LDA+U with ldaplusu_j != 0 is not implemented on the accelerated path: "
            "PARSEC's J term gives a potential matrix that is not symmetric; run "
            "the input with parsec_python/reference_main.py"
        )
    if hubbard and mpi_context is not None:
        raise NotImplementedError(
            "LDA+U is not implemented for the MPI SCF of several ranks; run it in "
            "one process (--serial-control)"
        )
    if (
        hubbard
        and is_periodic
        and getattr(getattr(problem, "grid", None), "periodicity", 3) in (1, 2)
    ):
        raise NotImplementedError(
            "LDA+U of a wire or slab is not implemented on the accelerated path; run "
            "the input with parsec_python/reference_main.py"
        )


def profile_hamiltonian_components(
    system: AcceleratedPreparedSinglePointSystem,
    *,
    block_size: int | None = None,
    repeats: int = 1,
    random_seed: int = 19,
) -> dict[str, float]:
    """Benchmark the three matrix-free Hamiltonian actions explicitly.

    This opt-in diagnostic synchronizes between components, so it is kept out
    of the production Chebyshev recurrence.  The diagonal field is set to the
    local ionic potential, making the reported local action specifically the
    requested ``V_ion,local`` benchmark.  Initial CA-LDA construction is timed
    separately by the reference SCF timing model.
    """

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if block_size is None:
        block_size = system.input.eigensolver.matvec_block_size
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")

    previous = getattr(system.backend, "local_potential", None)
    previous = None if previous is None else np.asarray(previous).copy()
    system.backend.bind(system.ionic_potential)
    generator = np.random.default_rng(random_seed)
    vectors = generator.standard_normal((system.grid.size, block_size))

    # One unreported warm-up prevents import/allocation/JIT startup from being
    # confused with a physical component cost in the optional microprofile.
    system.backend.synchronize()
    started = perf_counter()
    system.backend.apply_kinetic(vectors)
    system.backend.synchronize()
    system.backend.statistics.warmup_seconds += perf_counter() - started

    totals: dict[str, float] = {}
    for _ in range(repeats):
        sample = system.backend.profile_components(vectors)
        for name, value in sample.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    averages = {name: value / repeats for name, value in totals.items()}
    system.backend.statistics.component_profile_seconds = dict(averages)
    if previous is not None:
        system.backend.update_local(previous)
    return averages


#: The peak of the full grid of both spin channels on one device (the fallback of a spin sector run, ``off`` of
#: ``PARSEC_SPIN_FALLBACK_SHARED``) in units of the two bases (states and subspace buffer at every grid point, FP64):
#: the neutral vacancy in 5^3 diamond cubes (729,000 points, 2004 + 6 states a channel, 21.8 GiB) held 77.4 GiB after
#: its second iteration and stopped in the third for 11.7 GB more on an 80 GB A100 (2026-11-19, jvf7_fullgrid65), at
#: least 4.3 times; 4.0 is counted, so that a fallback that may still fit is tried.
SPIN_FALLBACK_PEAK_FACTOR = 4.0


class SpinFallbackPlan:
    """Where the full grid of both channels of a spin sector run would go if its guard fired (:func:`spin_fallback_plan`).

    ``basis_bytes``: the two channels' bases on the full grid; ``device_bytes``: the memory of the device that holds
    them in the layout of before (the first GPU of the process, of the root rank in the runner); ``devices``: the GPUs
    of the process or the ranks of the runner that could share them; ``divided``: whether the first solve divides a
    shared basis over them (CHEBFF); ``shareable``: a semilocal run (a hybrid's exchange and a meta-GGA's orbital term
    keep one device per sector basis); ``setting``: ``PARSEC_SPIN_FALLBACK_SHARED``.  ``layout`` is ``shared`` (one
    sector of the identity shared by the devices or ranks) or ``device`` (the full grid of one device, as before).
    ``refuses``: a fallback that fits nowhere stops before its full grid is prepared, under ``auto`` and ``on``;
    ``off`` tries the full grid of one device whatever the estimate, as before.  ``moves`` names a plan of a spin
    cell that the command line solves on the full grid from its start (molecular dynamics, manual movements,
    ``Move_Flag``, ``Ignore_Symmetry``: :func:`spin_full_grid_plan`) rather than a fallback.
    The root alone builds and acts on the plan of a fallback; the other ranks of the runner follow its commands.
    """

    def __init__(
        self,
        *,
        basis_bytes,
        device_bytes,
        devices,
        divided,
        shareable,
        setting,
        mpi,
        moves=False,
        rank_devices=1,
        gpu_per_rank=True,
    ) -> None:
        self.basis_bytes = int(basis_bytes)
        self.device_bytes = None if device_bytes is None else int(device_bytes)
        self.devices = int(devices)
        self.divided = bool(divided)
        self.shareable = bool(shareable)
        self.setting = str(setting)
        self.mpi = bool(mpi)
        self.moves = bool(moves)
        # The GPUs of each rank of the runner (one rank per node: those of its node) and whether the launch gives
        # each rank one GPU (--rank-per-gpu): a shared layout on ranks of several GPUs each holds each rank's share
        # on its first GPU alone (the layout PARSEC_SINGLE_SECTOR_NODE_LAYOUT refuses elsewhere; the fallback takes
        # it, and the NOTE of the preparation says so).
        self.rank_devices = int(rank_devices)
        self.gpu_per_rank = bool(gpu_per_rank)

    @property
    def nested(self) -> bool:
        """Whether a shared layout of the runner spreads each rank's share over the several GPUs of its node.

        The fallback's ``reprepare`` configures its sector of the identity as the runner configures
        any shared sector, so nested sharing takes it where it applies (``PARSEC_MPI_NESTED_SHARING`` other than
        ``off``, a CHEBFF first solve, one rank per node of several GPUs: ``experimental/mpi_scf.
        nested_sharing_applies``); under ``off`` the plan of before.
        """

        if not self.mpi:
            return False
        from .experimental.mpi_scf import nested_sharing_applies

        return nested_sharing_applies(self.devices, self.rank_devices, gpu_per_rank=self.gpu_per_rank,
                                      exchange=not self.shareable, divided=self.divided)

    @property
    def first_gpu_shares(self) -> bool:
        """Whether a shared layout of the runner would hold each rank's share on the first of its several GPUs."""

        return self.mpi and not self.gpu_per_rank and self.rank_devices > 1 and not self.nested

    @property
    def estimate_bytes(self) -> int:
        return int(SPIN_FALLBACK_PEAK_FACTOR * self.basis_bytes)

    @property
    def fits_one_device(self) -> bool:
        return self.device_bytes is None or self.estimate_bytes <= self.device_bytes

    @property
    def sharing_devices(self) -> int:
        """The devices among which a shared layout divides the bases (1 where the first solve does not divide them;
        every GPU of every rank under nested sharing, :attr:`nested`)."""

        if not self.divided:
            return 1
        return self.devices * self.rank_devices if self.nested else self.devices

    @property
    def layout(self) -> str:
        if self.setting == "off" or not self.shareable or self.devices < 2:
            return "device"
        if self.setting == "on":
            return "shared"
        return "device" if self.fits_one_device or self.sharing_devices < 2 else "shared"

    @property
    def fits(self) -> bool:
        if self.device_bytes is None:
            return True
        if self.layout == "device":
            return self.fits_one_device
        return self.estimate_bytes <= self.device_bytes * self.sharing_devices

    @property
    def refuses(self) -> bool:
        """Whether a fallback of this plan stops before its full grid is prepared (it fits nowhere, not ``off``)."""

        return not self.fits and self.setting != "off"

    def _where(self) -> str:
        return f"the {self.devices} ranks of the runner" if self.mpi else f"the {self.devices} GPUs of this process"

    def layout_text(self) -> str:
        """Where the full grid of a fallback goes, for its NOTE in parsec.out (``none`` where the run stops)."""

        if self.refuses:
            return "none"
        if self.layout == "shared":
            return (
                f"as one sector of the identity whose basis {self._where()} share "
                + (
                    "(each on the first GPU of its node) " if self.first_gpu_shares
                    else "(each over the GPUs of its node, PARSEC_MPI_NESTED_SHARING) " if self.nested
                    else ""
                )
                + f"(PARSEC_SPIN_FALLBACK_SHARED={self.setting})"
            )
        return "on the full grid of one device (Spin_Symmetry_Sectors: off)"

    def text(self) -> str:
        """What the plan says about the full grid of the fallback (GiB), for the NOTE and for the error."""

        gib = 2.0**30
        head = (
            ("the full grid of both spin channels of this cell (molecular dynamics, manual movements, Move_Flag or "
             "Ignore_Symmetry use no point group)" if self.moves else "the full grid of both spin channels")
            + f" would take an estimated {self.estimate_bytes / gib:.1f} GiB "
            f"({self.basis_bytes / gib:.1f} GiB of orbitals times {SPIN_FALLBACK_PEAK_FACTOR:g})"
        )
        one = "" if self.device_bytes is None else f" against {self.device_bytes / gib:.1f} GiB of one device"
        if self.layout == "shared":
            share = (
                f"; the run then takes one sector of the identity shared by {self._where()}"
                + (f" over their {self.sharing_devices} GPUs (PARSEC_MPI_NESTED_SHARING)" if self.nested else "")
                + (f", about {self.estimate_bytes / self.sharing_devices / gib:.1f} GiB each" if self.divided else
                   ", each of which holds the whole basis after a first solve other than CHEBFF")
                + f" (PARSEC_SPIN_FALLBACK_SHARED={self.setting})"
                + (
                    f"; each rank holds its share on the first of its {self.rank_devices} GPUs (one rank per node), so "
                    f"{self.devices} of the {self.devices * self.rank_devices} GPUs hold the basis: "
                    "scripts/run_multi_gpu_scf.sh --rank-per-gpu shares it over every GPU"
                    if self.first_gpu_shares
                    else ""
                )
            )
            return head + one + share + ("" if self.fits else ": that does not fit either")
        why = (
            "PARSEC_SPIN_FALLBACK_SHARED=off" if self.setting == "off"
            else "a hybrid or meta-GGA keeps one device per basis" if not self.shareable
            else "one GPU" if self.devices < 2
            else "a first solve other than CHEBFF does not divide a shared basis"
        )
        return head + one + (", one device" if self.moves else ", the layout of before on one device") + f" ({why})"

    def note(self) -> str | None:
        """The NOTE of the preparation where the fallback would not fit one device, or would share its full grid on
        the first GPUs of ranks of several GPUs each (:attr:`first_gpu_shares`), ``None`` elsewhere."""

        if self.fits_one_device and not (self.layout == "shared" and self.first_gpu_shares):
            return None
        return (
            "should the symmetry guard of the spin sectors fire (Spin_Symmetry_Sectors), "
            + self.text()
            + (
                ""
                if self.fits
                else "; such a run stops with an error before that full grid is prepared "
                "(PARSEC_SPIN_FALLBACK_SHARED=off tries the full grid of one device anyway)"
                if self.refuses
                else "; it is tried anyway, as before"
            )
        )

    def moves_note(self) -> str | None:
        """The NOTE of the command line for a spin cell it solves on the full grid (``moves``), ``None`` where the
        full grid of both channels fits one device: what was decided (the shared layout, or the 1x1x1 k-point driver
        of the reference solver where no layout fits)."""

        if self.fits_one_device and self.layout == "device":
            return None
        if self.fits:
            return self.text()
        return (
            self.text() + ("" if self.layout == "shared" else ": that does not fit")
            + "; the run therefore keeps the 1x1x1 k-point driver of the reference solver "
            "(PARSEC_SPIN_SECTOR_FORCES=off keeps it for every such run)"
        )


def spin_fallback_plan(reference, *, mpi_context=None) -> SpinFallbackPlan:
    """The :class:`SpinFallbackPlan` of a spin-polarized run in the sectors, from its prepared ``reference``.

    The device memory is the total of the first GPU of the process (``PARSEC_CUPY_DEVICES``), which the layout of
    before fills; the bytes of a shared layout are divided by the devices of the process, or by the ranks of the
    runner (one GPU each for a shared sector), where a CHEBFF first solve divides the basis over them.
    """

    from parsec_python.SCF.single_point import _number_of_states

    states = int(_number_of_states(reference)) + int(reference.input.eigensolver.subspace_buffer)
    basis = 2 * int(reference.grid.size) * states * np.dtype(np.float64).itemsize
    return _spin_plan(basis, reference.input.scf, reference.input.eigensolver, mpi_context=mpi_context)


def spin_full_grid_plan(problem) -> SpinFallbackPlan | None:
    """The :class:`SpinFallbackPlan` (``moves``) of a spin-polarized bulk cell that the command line would solve on
    the full grid of the accelerated driver from its start, from the input alone (before anything is prepared).

    Molecular dynamics, manual movements, ``Move_Flag`` fixing atoms and ``Ignore_Symmetry`` take no point group
    (:func:`.SCF.gamma_spin.spin_full_grid_moves_plan`): both channels on the full grid of one device, or, where they
    would not fit there, one sector of the identity that the GPUs of the process share (``PARSEC_SPIN_FALLBACK_SHARED``),
    or, where that does not fit either, the 1x1x1 k-point driver of before.  The grid is that of the default routines
    (:func:`parsec_python.Grid.pbc.default_periodic_grid_counts`), the states ``States_Num`` (or half the valence
    electrons and six) and the subspace buffer.  ``None`` where the input does not tell (no estimate: the run goes on
    as without one).
    """

    if getattr(problem, "periodic_cell", None) is None or not getattr(problem.scf, "spin_polarized", False):
        return None
    try:
        from parsec_python.Grid.pbc import default_periodic_grid_counts

        points = int(np.prod(default_periodic_grid_counts(problem.periodic_cell, problem.grid)))
        states = problem.scf.number_of_states
        if states is None:
            from parsec_python.V_ion.ionic_potential import ionic_charge, load_pseudopotentials

            potentials = load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
            electrons = ionic_charge(problem.atoms, potentials) - problem.scf.net_charge
            states = int(np.ceil(0.5 * electrons)) + 6
        states = int(states) + int(problem.eigensolver.subspace_buffer)
    except Exception:
        return None
    basis = 2 * points * states * np.dtype(np.float64).itemsize
    return _spin_plan(basis, problem.scf, problem.eigensolver, mpi_context=None, moves=True)


def _spin_plan(basis, scf, eigensolver, *, mpi_context=None, moves=False) -> SpinFallbackPlan:
    """The :class:`SpinFallbackPlan` of ``basis`` bytes: the devices of the process (or the ranks of the runner)."""

    from .SCF.spin_sectors import spin_fallback_shared_setting

    device_bytes, devices = None, 1
    try:
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        ids = selected_device_ids(int(cp.cuda.Device().id))
        with cp.cuda.Device(ids[0]):
            device_bytes = int(cp.cuda.runtime.memGetInfo()[1])
        devices = len(ids)
    except Exception:
        pass
    rank_devices = devices
    if mpi_context is not None:
        devices = int(getattr(mpi_context, "size", 1))
    return SpinFallbackPlan(
        basis_bytes=basis,
        device_bytes=device_bytes,
        devices=devices,
        divided=str(getattr(eigensolver, "method", "")).strip().lower() == "chebff",
        shareable=not (getattr(scf, "is_hybrid", False) or getattr(scf, "is_meta_gga", False)
                       or bool(getattr(mpi_context, "exchange", False))),
        setting=spin_fallback_shared_setting(),
        mpi=mpi_context is not None,
        moves=moves,
        rank_devices=rank_devices if mpi_context is not None else 1,
        gpu_per_rank=bool(getattr(mpi_context, "gpu_per_rank", True)) if mpi_context is not None else True,
    )


def _weighted_sectors(sectors) -> bool:
    """Whether the sector eigensolver ``sectors`` (or ``None``) solves the weighted sectors of a subgroup."""

    from .Symmetry import PointGroupReduction

    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    return isinstance(reduction, PointGroupReduction)


def _keep_spin_flip_left(system) -> None:
    """A system prepared again in the place of one whose run left ``PARSEC_SPIN_FLIP_SYMMETRY`` keeps both channels.

    The mark of :func:`parsec_python.SCF.spin_polarized.leave_spin_flip` stays on the system object that
    :class:`_SpinSectorFallback` fills; this corrects the detail ``spin_flip_symmetry`` that the new preparation wrote
    (the route of ``off`` from there on, as the run had it).
    """

    from parsec_python.SCF.spin_polarized import SPIN_FLIP_LEFT, leave_spin_flip

    reason = getattr(system, SPIN_FLIP_LEFT, None)
    if reason:
        leave_spin_flip(system, reason)


class _SpinSectorFallback:
    """How a spin-polarized run in the symmetry sectors goes back to the full grid.

    A run in the weighted sectors of a subgroup (``Symmetry/weighted_inputs.py``) whose spin guard fires, or whose
    ``occup.in`` fills the partners of a weighted state unequally (``spin=False``: an unpolarized run with
    ``occup.in``, which has no guard), is first prepared again in the sectors of the abelian group
    (:meth:`replace_with_abelian`; on every rank of ``mpi_context``, which the root tells to prepare them with it),
    whose own guard may still send a spin run to the full grid.  A spin guard whose report the abelian sectors
    would repeat on the same levels (``FractionalOccupations.abelian``: a multiplet with partners in two or more of
    their sectors; a cluster, a cell whose operations take no translation, and since 2026-11-22 a cell whose abelian
    sectors take its commuting involutions with their translations, ``PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS``)
    goes straight to the full grid under
    ``PARSEC_WEIGHTED_GUARD_SHORTCUT=auto``: those sectors would solve the same state again and stop it there
    (``off``: through them, as before).  At the start Hamiltonian (test 1), which both share, that is exact, and the
    result is the abelian route's bit for bit; at the end of the SCF (test 3) it is predicted from the weighted
    occupations, and the full grid gives the reference without symmetry, the abelian route's result wherever that
    route stops at its end too.

    ``Spin_Symmetry_Sectors`` keeps a sector result only where the symmetry
    guard of :mod:`.SCF.spin_sectors` passes.  Otherwise :meth:`run`
    prepares the calculation again, from the input as it was given, with
    ``Spin_Symmetry_Sectors: off`` and the atoms of the present structure,
    puts the full-grid objects in place of the sector ones in the very
    system the caller holds (forces, outputs and the next ionic steps then
    use them), says so to the iteration callback and runs the SCF there:
    from the atomic densities, the full-grid run of that structure as
    ``off`` would have given it, bit for bit; after an ionic move from the
    potentials the sector run started from (:meth:`full_grid_warm_start`),
    as an ``off`` relaxation starts from its previous structure, with fresh
    subspaces (equal to ``off`` within the SCF tolerance, not bit for bit:
    the previous structures were sector runs and ``off`` keeps its
    subspaces).  The pseudopotential files are read here,
    while they exist, so that the full grid can be prepared from them
    later even where the caller has removed them (a temporary directory).

    ``plan`` (:class:`SpinFallbackPlan`, ``PARSEC_SPIN_FALLBACK_SHARED``)
    says where that full grid goes: the first device of the process as
    above, or, where both channels would not fit there, one sector of the
    identity whose basis the GPUs of the process share (``prepare_single_point``
    with ``spin_identity_sector``), or the ranks of ``mpi_context``, which the
    root tells to prepare it with it (:meth:`MPISectorContext.reprepare`);
    a plan that fits nowhere stops the run before anything is prepared.
    """

    def __init__(
        self,
        problem,
        atoms,
        *,
        backend,
        symmetry,
        symmetry_cache_directory,
        mpi_context=None,
        plan=None,
        symmetry_fixed_atoms=None,
        symmetry_constraints=(),
        spin: bool = True,
        sector_forces: bool = False,
    ) -> None:
        self.problem = problem
        self.spin = bool(spin)
        # (the forces and moves of a cell averaged over the group of its sectors: the abelian sectors prepared again
        # keep that, as the weighted ones were prepared, prepare_single_point)
        self.sector_forces = bool(sector_forces)
        self.atoms = tuple(atoms)
        self.backend = backend
        self.symmetry = symmetry
        self.symmetry_cache_directory = symmetry_cache_directory
        # The sector context of the runner (its ranks follow a shared layout) and the SpinFallbackPlan.
        self.mpi_context = mpi_context
        self.plan = plan
        # (the fixed atoms of a relaxation that keeps the subgroup of its Move_Flag: the full grid prepared again finds
        # the same subgroup, which the moved atoms keep)
        self.symmetry_fixed_atoms = symmetry_fixed_atoms
        # (the points the operations must keep besides the atoms, a field axis of Polarizability:
        # SCF/symmetry_off_inputs.py; the abelian sectors prepared again keep them)
        self.symmetry_constraints = tuple(symmetry_constraints)
        self.pseudopotential_files = {}
        for symbol, specification in getattr(problem, "pseudopotentials", {}).items():
            try:
                self.pseudopotential_files[symbol] = Path(specification.path).read_bytes()
            except OSError:
                pass

    def run(self, system, run, *, callback=None, warm_start=None, **keywords):
        """``run(system, ...)`` in the sectors; again on the full grid where the guard says so."""

        from parsec_python.Occupations import UnequalPartnerOccupations

        from .SCF.hybrid import WeightedHybridAbandoned
        from .SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        while True:
            abandoned = None
            # (the record of a hybrid that left the weighted sectors names the hybrid's reason)
            route = "weighted"
            weighted = _weighted_sectors(getattr(system.backend, "symmetry_eigensolver", None))
            try:
                result = run(system, callback=callback, warm_start=warm_start, **keywords)
                if self.spin:
                    system.scalar_field_adapter.check_spin_result(result, final=True)
                    self._say_doublet(system, callback)
                return result
            except SpinSectorsAbandoned as error:
                # Only the report leaves this block: the frames of the sector run,
                # and the orbitals they hold, go with the exception.
                abandoned = (error.stage, error.report)
            except UnequalPartnerOccupations as error:
                if not weighted:
                    raise
                abandoned = (
                    "the occupations of occup.in",
                    FractionalOccupations(
                        largest=float(np.ptp(error.entries)),
                        tolerance=0.0,
                        lines=(str(error),),
                        reason="unequal occupations of degenerate partners",
                        measure="largest difference of the occupations of the partners",
                    ),
                )
            except WeightedHybridAbandoned as error:
                # an unpolarized hybrid with a partly filled degenerate level (SCF/hybrid.py): the abelian sectors,
                # where the exact exchange can split it
                if not weighted:
                    raise
                abandoned = (error.stage, error.report)
                route = "weighted_hybrid"
            if not weighted:
                break
            stage, report = abandoned
            # PARSEC_WEIGHTED_GUARD_SHORTCUT=auto: the abelian sectors would stop the same state (see the class).
            # (a doublet within one sector: the abelian sectors hold it in one sector too and would keep the state)
            doublet = bool(getattr(report, "doublet", False))
            skip_abelian = self.spin and (doublet or self.abelian_shortcut(report))
            if callback is not None:
                from parsec_python.models import SpinSectorFallbackRecord

                callback(
                    SpinSectorFallbackRecord(
                        stage=stage,
                        largest_fractional_occupation=report.largest,
                        tolerance=report.tolerance,
                        multiplets=report.lines,
                        reason=report.reason,
                        measure=report.measure,
                        route=route,
                        **({"abelian_skipped": True} if skip_abelian else {}),
                        **({"doublet": True} if doublet else {}),
                    )
                )
            if skip_abelian:
                break
            self.replace_with_abelian(system, stage, report)
            warm_start = self.full_grid_warm_start(warm_start)
        stage, report = abandoned
        if callback is not None:
            # Said before the full-grid preparation, which may fail on its own.
            from parsec_python.models import SpinSectorFallbackRecord

            callback(
                SpinSectorFallbackRecord(
                    stage=stage,
                    largest_fractional_occupation=report.largest,
                    tolerance=report.tolerance,
                    multiplets=report.lines,
                    reason=report.reason,
                    measure=report.measure,
                    # where the full grid goes (PARSEC_SPIN_FALLBACK_SHARED): named in the NOTE of parsec.out
                    **({} if self.plan is None else {"layout": self.plan.layout_text()}),
                )
            )
        self.replace_with_full_grid(system, stage, report)
        return run(system, callback=callback, warm_start=self.full_grid_warm_start(warm_start), **keywords)

    @staticmethod
    def _say_doublet(system, callback) -> None:
        """The NOTE of a kept result with a fractional doublet within one sector (``PARSEC_SPIN_GUARD_DOUBLETS=note``)."""

        noted = getattr(system.scalar_field_adapter, "doublet_report", None)
        if noted is None or callback is None:
            return
        from parsec_python.models import SpinSectorFallbackRecord

        from .SCF.spin_sectors import SECTOR_DOUBLET_REASON

        stage, report = noted
        callback(
            SpinSectorFallbackRecord(
                stage=stage,
                largest_fractional_occupation=report.largest,
                tolerance=report.tolerance,
                multiplets=report.lines,
                reason=report.reason,
                measure=report.measure,
                # (levels of one label of the abelian sectors, SCF.spin_sectors.fractional_sector_doublets)
                route="sector_doublet" if report.reason == SECTOR_DOUBLET_REASON else "doublet",
                doublet=True,
            )
        )

    @staticmethod
    def abelian_shortcut(report) -> bool:
        """Whether a guard ``report`` of the weighted sectors goes straight to the full grid
        (``PARSEC_WEIGHTED_GUARD_SHORTCUT=auto`` and the abelian sectors would report the same levels:
        ``FractionalOccupations.abelian``, set by the guard of the weighted sectors of a cluster, of a cell whose
        operations take no translation, or of a cell with translations under
        ``PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS=auto``)."""

        from .Symmetry.weighted_inputs import guard_shortcut_setting

        return bool(getattr(report, "abelian", False)) and guard_shortcut_setting() == "auto"

    @staticmethod
    def full_grid_warm_start(warm_start):
        """The warm start of the full-grid run that replaces a sector run started from ``warm_start``.

        ``None`` (a run from the atomic densities) stays ``None``.  After an
        ionic move the potentials of both channels and the Hartree potential
        of the previous structure are full-grid fields already (the public
        fields of a sector result, a meta-GGA's ``vtau`` and ``W`` included)
        and are kept, as an ``off`` relaxation keeps them; what lives in the
        sectors is dropped: the eigensolver
        subspaces (the full grid starts with its first filter), a hybrid's
        exchange orbitals and a carried mixer.
        """

        if warm_start is None:
            return None
        return replace(
            warm_start,
            eigensolver_state=None,
            exchange_orbitals=None,
            mixer=None,
            first_iteration_degree=None,
        )

    def _problem_of(self, system, *, shared: bool = False, full_grid: bool = True):
        """The input of the full-grid run: as given, at the atoms of the present structure.

        ``Spin_Symmetry_Sectors: off`` for the full grid of one device; as given for the shared layout, whose one
        sector of the identity the driver prepares without a guard (``spin_identity_sector``), and for the run in the
        abelian sectors that replaces weighted ones (``full_grid=False``).
        """

        scf = (
            replace(self.problem.scf, spin_symmetry_sectors="off")
            if full_grid and not shared
            else self.problem.scf
        )
        atoms = tuple(system.reference.atoms)
        if _same_atoms(atoms, self.atoms):
            return replace(self.problem, scf=scf)
        # A moved structure: its atoms are in the frame of the grid already.
        return replace(self.problem, scf=scf, atoms=atoms, recenter_geometry=False)

    def _with_present_pseudopotentials(self, problem, directory: Path):
        """``problem`` with a copy in ``directory`` of each pseudopotential file that is gone since."""

        species = dict(problem.pseudopotentials)
        changed = False
        for symbol, specification in species.items():
            path = Path(specification.path)
            if path.exists() or symbol not in self.pseudopotential_files:
                continue
            copy = directory / symbol / path.name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(self.pseudopotential_files[symbol])
            species[symbol] = replace(specification, path=copy)
            changed = True
        return replace(problem, pseudopotentials=species) if changed else problem

    def replace_with_abelian(self, system, stage: str, report) -> None:
        """Prepare the system in the sectors of the abelian group and put its objects into ``system``.

        What the weighted sectors hold on the devices goes first.  The preparation keeps the abelian group of the
        orbitals (:func:`.Symmetry.weighted_inputs.abelian_orbitals`) and says why in its backend details.  With an
        ``mpi_context`` (the full-SCF runner, ``PARSEC_ORBITAL_GROUP_MPI=auto``) the root first tells the other ranks
        to prepare the same input in the abelian sectors (:meth:`.experimental.mpi_scf.MPISectorContext.reprepare`),
        so that every rank configures the sectors of the same plan again, and its own densities are summed over the
        ranks, as the runner set up those of the first preparation.
        """

        import gc
        from dataclasses import fields
        import tempfile

        from .Symmetry.weighted_inputs import abelian_orbitals

        problem = self._problem_of(system, full_grid=False)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        if sectors is not None:
            if self.spin and getattr(sectors, "spin_channel_count", 1) == 2:
                for index in (0, 1):
                    sectors.for_spin_channel(index).reset()
            else:
                sectors.reset()
        gc.collect()
        reason = f"the weighted sectors were left after {stage} ({report.reason})"
        context = self.mpi_context
        options = dict(
            backend=self.backend,
            symmetry=self.symmetry,
            symmetry_cache_directory=self.symmetry_cache_directory,
            # (the abelian group of a relaxation that keeps the subgroup of its Move_Flag is detected with the
            # fixed atoms labelled too, PARSEC_MOVE_FLAG_SUBGROUP)
            symmetry_fixed_atoms=self.symmetry_fixed_atoms,
            symmetry_constraints=self.symmetry_constraints,
            **({"sector_forces": True} if self.sector_forces else {}),
        )
        if context is not None:
            # The ranks of the runner prepare the abelian sectors together: the root sends them the input first (with
            # the pseudopotential files of the input, which every rank read at its start), and each replaces the
            # sector solver of its worker loop.
            context.reprepare(dict(problem=problem, abelian_reason=reason, **options))
        with tempfile.TemporaryDirectory() as copies, abelian_orbitals(reason):
            problem = self._with_present_pseudopotentials(problem, Path(copies))
            abelian = prepare_single_point(
                problem,
                **options,
                **({"mpi_context": context} if context is not None else {}),
            )
        if context is not None:
            # The root's densities of the abelian sectors are summed over the ranks, as the runner set it up.
            from .experimental.mpi_scf import MPISymmetryDensityBuilder

            abelian.orbital_density_builder = MPISymmetryDensityBuilder(
                abelian.orbital_density_builder, context, abelian.backend.symmetry_eigensolver
            )
        note = (
            "weighted_sectors",
            f"left after {stage} ({report.reason}: {report.measure} = {report.largest:.3e}): the sectors of the "
            "abelian group",
        )
        abelian.backend_info = replace(
            abelian.backend_info,
            details=tuple(item for item in abelian.backend_info.details if item[0] != note[0]) + (note,),
        )
        abelian.backend.info = abelian.backend_info
        kept = system.materialize_final_wavefunctions
        for item in fields(abelian):
            setattr(system, item.name, getattr(abelian, item.name))
        system.materialize_final_wavefunctions = kept
        _keep_spin_flip_left(system)
        del abelian, sectors
        gc.collect()
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()

    def replace_with_full_grid(self, system, stage: str, report) -> None:
        """Prepare the full-grid system and put its objects into ``system``."""

        import gc
        from dataclasses import fields

        plan = self.plan
        shared = plan is not None and plan.layout == "shared"
        if plan is not None and plan.refuses:
            # Decided before anything is freed or prepared: the full grid would run out of device memory (the
            # preparation said so in its NOTE).  Under PARSEC_SPIN_FALLBACK_SHARED=off the full grid of one device is
            # tried whatever the estimate, as before.
            raise ValueError(
                f"Spin_Symmetry_Sectors: the run left the symmetry sectors after {stage} ({report.reason}), but "
                f"{plan.text()}: this run has no full grid that fits. Run it on more GPUs (the full-SCF runner, "
                "scripts/run_multi_gpu_scf.sh --rank-per-gpu, with PARSEC_SPIN_FALLBACK_SHARED=auto shares that full "
                "grid among its ranks), or with Spin_Symmetry_Sectors: off on a route that holds it "
                "(PARSEC_SPIN_FALLBACK_SHARED=off tries the full grid of one device anyway)"
            )
        problem = self._problem_of(system, shared=shared)
        # What the sectors hold on the devices goes first: the subspaces of
        # both channels and the exchange's plans.
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        if sectors is not None and getattr(sectors, "spin_channel_count", 1) == 2:
            for index in (0, 1):
                sectors.for_spin_channel(index).reset()
        exchange = getattr(system, "exchange_device", None)
        if exchange is not None:
            exchange.bind(None)
            # On several MPI ranks: what every rank's exchange service holds
            # (replicas, W, the window of the outer DIIS, the device plans of
            # its pair solvers) goes before the root prepares the full grid.
            release = getattr(exchange, "release_build_memory", None)
            if callable(release):
                release(final=True)
            exchange._solvers.clear()
            exchange.expansions.clear()
        gc.collect()
        import tempfile

        context = self.mpi_context if shared else None
        if shared and context is not None:
            # The ranks of the runner prepare the shared full grid together: the root sends them the input first
            # (MPISectorContext.reprepare), and each replaces the sector solver of its worker loop.
            context.reprepare(
                dict(problem=problem, backend=self.backend, symmetry_cache_directory=self.symmetry_cache_directory)
            )
        try:
            with tempfile.TemporaryDirectory() as copies:
                problem = self._with_present_pseudopotentials(problem, Path(copies))
                full_grid = prepare_single_point(
                    problem,
                    backend=self.backend,
                    # ``on`` asked for the sectors, which were taken; the full grid
                    # of two spin channels has no reduced component ``on`` demands.
                    symmetry=(
                        "off"
                        if shared
                        else "auto" if _normalize_symmetry_mode(self.symmetry) == "on" else self.symmetry
                    ),
                    symmetry_cache_directory=self.symmetry_cache_directory,
                    # PARSEC_SPIN_FALLBACK_SHARED: one sector of the identity shared by the GPUs or the ranks
                    **({"spin_identity_sector": True} if shared else {}),
                    **({"mpi_context": context} if context is not None else {}),
                    symmetry_fixed_atoms=self.symmetry_fixed_atoms,
                )
        except ValueError as error:
            # PARSEC_HARTREE_SYMMETRY=1 forbids the full grid of a spin run.
            raise ValueError(
                f"Spin_Symmetry_Sectors: the run left the symmetry sectors after {stage} "
                f"({report.reason}), and the full grid cannot be prepared: {error}"
            ) from error
        note = (
            "spin_symmetry_sectors",
            f"auto: left the sectors after {stage} ({report.reason}: {report.measure} = "
            f"{report.largest:.3e} >= {report.tolerance:.0e}): full grid"
            + (
                ", one sector of the identity shared by "
                + (f"{plan.devices} MPI ranks" if context is not None else f"{plan.devices} GPUs")
                + f" (PARSEC_SPIN_FALLBACK_SHARED={plan.setting})"
                if shared
                else ""
            ),
        )
        if context is not None:
            # The root's densities of the shared full grid are summed over the ranks, as the runner set it up.
            from .experimental.mpi_scf import MPISymmetryDensityBuilder

            full_grid.orbital_density_builder = MPISymmetryDensityBuilder(
                full_grid.orbital_density_builder, context, full_grid.backend.symmetry_eigensolver
            )
        elif not shared and self.spin and self.mpi_context is None:
            # PARSEC_SPIN_FALLBACK_DENSITY=abelian: the channel densities keep the abelian group of the atoms
            # (SCF/fallback_density.py); off (the default) keeps the full grid without symmetry.
            from .SCF.fallback_density import with_fallback_density

            pinned = with_fallback_density(full_grid)
            if pinned is not None:
                note = (note[0], f"{note[1]}, {pinned}")
        full_grid.backend_info = replace(
            full_grid.backend_info,
            details=tuple(item for item in full_grid.backend_info.details if item[0] != note[0]) + (note,),
        )
        full_grid.backend.info = full_grid.backend_info
        kept = system.materialize_final_wavefunctions
        for item in fields(full_grid):
            setattr(system, item.name, getattr(full_grid, item.name))
        system.materialize_final_wavefunctions = kept
        _keep_spin_flip_left(system)
        del full_grid, sectors, exchange
        gc.collect()
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()


def _say_floor_doublet(system, result, callback) -> None:
    """The NOTE of a spin cell that its size floor keeps on the full grid and that ends with a partly filled pair of
    degenerate levels (``spin_floor_full_grid`` of the preparation; only a process without MPI sectors has it).  It
    reads the final levels and changes nothing."""

    if callback is None or not getattr(system, "spin_floor_full_grid", False):
        return
    channels = getattr(getattr(result, "result", result), "spin", None)
    if channels is None:
        return
    from parsec_python.models import SpinSectorFallbackRecord

    from .SCF.spin_sectors import partly_filled_doublets

    report = partly_filled_doublets(channels)
    if report is None:
        return
    callback(
        SpinSectorFallbackRecord(
            stage="the SCF loop",
            largest_fractional_occupation=report.largest,
            tolerance=report.tolerance,
            multiplets=report.lines,
            reason=report.reason,
            measure=report.measure,
            route="floor_doublet",
            doublet=True,
        )
    )


def _say_symmetric_level(system, result, callback) -> None:
    """The NOTE of an unpolarized run in the weighted sectors that ends with a partly filled degenerate level at the
    Fermi level (Fermi-Dirac occupations, not ``occup.in``): the sectors keep the density of the full group, so every
    partner of the level holds one occupation, while a run without that constraint may reach a broken-symmetry state
    that lies lower or higher (:func:`.SCF.spin_sectors.weighted_partly_filled_levels`).  It reads the final levels
    and changes nothing.  Not for a hybrid: the guard of its Fock loop (``SCF/hybrid.py``, ``weighted_guard``) reads
    the same levels and starts such a run again in the abelian sectors (NOTE route ``weighted_hybrid``)."""

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    if callback is None or not _weighted_sectors(sectors):
        return
    inner = getattr(result, "result", result)
    if getattr(inner, "spin", None) is not None:
        return
    scf = getattr(getattr(system, "input", None), "scf", None)
    if scf is None or getattr(scf, "spin_polarized", False) or getattr(scf, "fixed_occupations", None) is not None:
        return
    if getattr(scf, "is_hybrid", False):
        return
    from parsec_python.models import SpinSectorFallbackRecord

    from .SCF.spin_sectors import weighted_partly_filled_levels

    report = weighted_partly_filled_levels(inner)
    if report is None:
        return
    callback(
        SpinSectorFallbackRecord(
            stage="the SCF loop",
            largest_fractional_occupation=report.largest,
            tolerance=report.tolerance,
            multiplets=report.lines,
            reason=report.reason,
            measure=report.measure,
            route="symmetric_level",
            doublet=True,
        )
    )


def run_scf(
    system: AcceleratedPreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    warm_start=None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[..., None] | None = None,
) -> AcceleratedSinglePointResult:
    """Run the validated SCF loop using the selected execution backend.

    ``warm_start`` continues from the loop of the structure before an ionic
    move; :func:`warm_start_for_moved_atoms` builds it.
    ``convergence_criterion`` replaces the criterion of the input for this
    loop.  ``iteration_state_callback`` receives the fields of every
    iteration (``Save_Intermediate_Charge_Density``).

    A spin-polarized loop in the symmetry sectors (``Spin_Symmetry_Sectors``)
    whose occupations end fractional is run again on the full grid, and
    ``system`` holds the full-grid objects from then on
    (:class:`_SpinSectorFallback`).
    """

    keywords = dict(
        convergence_criterion=convergence_criterion,
        **(
            {}
            if iteration_state_callback is None
            else {"iteration_state_callback": iteration_state_callback}
        ),
    )
    fallback = getattr(system, "spin_sector_fallback", None)
    if fallback is None:
        result = run_accelerated_scf(
            system, callback=callback, warm_start=warm_start, **keywords
        )
        _say_floor_doublet(system, result, callback)
        _say_symmetric_level(system, result, callback)
    else:
        result = fallback.run(
            system, run_accelerated_scf, callback=callback, warm_start=warm_start, **keywords
        )
    # A hybrid on several MPI ranks: what the builds of its exchange left on
    # the devices of the ranks (replicas of other ranks' orbitals, the last
    # parts of W) goes before the forces and the next structure.
    release = getattr(getattr(system, "exchange_device", None), "release_build_memory", None)
    if callable(release):
        release()
    boundary_check = _hartree_boundary_check(system, result)
    if boundary_check:
        result.backend = replace(
            result.backend, details=result.backend.details + boundary_check
        )
    variational = getattr(system, "hartree_variational", None)
    if variational is not None:
        # The solves of A u = rho so far (PARSEC_HARTREE_VARIATIONAL), all structures of the run.
        result.backend = replace(
            result.backend,
            details=result.backend.details + (("hartree_variational_solves", variational.report()),),
        )
    solver = getattr(system, "hartree_solver", None)
    fallback = getattr(solver, "fallback", None)
    if isinstance(fallback, str):
        # The resident chain gave way to the route before during a solve.
        changed = dict(getattr(solver, "fallback_details", ()))
        changed.update(
            hartree_resident_chain=fallback,
            hartree_full_grid_transfer_policy="host-interface",
            hartree_resident_predictor="disabled",
        )
        result.backend = replace(
            result.backend,
            details=tuple(
                (name, changed.get(name, value))
                for name, value in result.backend.details
                if name != "hartree_resident_device_bytes"
            ),
        )
    return result


def update_geometry(
    system: AcceleratedPreparedSinglePointSystem, atoms
) -> AcceleratedPreparedSinglePointSystem:
    """Return the prepared system of moved atoms on the same grid.

    The counterpart of :func:`parsec_python.SCF.update_geometry`.  The
    Hamiltonian operators on the devices, the Hartree solver, the symmetry
    maps and the eigensolver with its saved subspaces are kept and shared
    with ``system``: their projectors, the atomic tail of the Hartree
    boundary and the core density are replaced in place, on every rank of an
    MPI run.  ``system`` itself keeps its static fields, which is what
    :func:`warm_start_for_moved_atoms` reads, but can no longer be solved.

    The symmetry sectors were chosen for the first structure.  The moved
    atoms must have every symmetry operation those sectors use; a relaxation
    keeps them by symmetrizing its forces.
    """

    updater = getattr(system, "geometry_updater", None)
    if updater is None:
        raise NotImplementedError(
            "this prepared system does not support moving its atoms"
        )
    return updater(system, atoms)


def warm_start_for_moved_atoms(
    previous_system, previous_result, system, *, extrapolation="none", history=None
):
    """What the SCF loop of moved atoms takes over, with ``extrapolation`` added.

    See :func:`parsec_python.SCF.warm_start_for_moved_atoms` for
    ``extrapolation``; the guessed density is evaluated on the devices by the
    SCF loop itself.
    """

    from parsec_python.SCF.single_point import extrapolated_warm_start, meta_gga_warm_start

    warm_start = meta_gga_warm_start(
        _parsec_warm_start(previous_system, previous_result, system), previous_result
    )
    exchange_device = getattr(system, "exchange_device", None)
    if exchange_device is not None and getattr(previous_result, "exact_exchange", None) is not None:
        # A hybrid functional: the orbitals before the move, where the
        # eigensolvers keep them, build the first exchange operator.
        warm_start = replace(
            warm_start, exchange_orbitals=exchange_device.orbital_sets(previous_result)
        )
    if getattr(system, "embedding", None) is not None:
        # FDET of PARSEC 1.5 (parsec_python.Embedding.warm_start): a translation
        # moves the subspace the device eigensolver keeps.
        from parsec_python.Embedding.warm_start import embedding_warm_start

        return embedding_warm_start(
            previous_system,
            previous_result,
            system,
            warm_start,
            translate_state=getattr(system.eigenproblem_solver, "translate_saved_rows", None),
        )
    return extrapolated_warm_start(
        warm_start,
        previous_system,
        previous_result,
        system,
        extrapolation,
        history,
    )


def _parsec_warm_start(previous_system, previous_result, system):
    """What the SCF loop of moved atoms takes over (``parsec.F90`` after ``domove``).

    The counterpart of :func:`parsec_python.SCF.warm_start_for_moved_atoms`:
    the last mixed potential with the local ionic potential exchanged, the
    last Hartree potential, and the eigensolver subspaces, which stay on
    their devices and ranks and are marked for a new loop there.
    """

    from parsec_python.SCF import SCFWarmStart, state_for_new_potential

    solver = system.eigenproblem_solver
    channels = getattr(previous_result, "spin", None)
    if channels is not None:
        # One potential and one saved subspace per spin channel.
        for_channel = getattr(solver, "for_spin_channel", None)
        return SCFWarmStart(
            input_potential=tuple(
                np.asarray(channel.next_effective_potential)
                - np.asarray(previous_system.ionic_potential)
                + np.asarray(system.ionic_potential)
                for channel in channels
            ),
            hartree_potential=np.asarray(previous_result.hartree_potential),
            eigensolver_state=tuple(
                for_channel(index).start_new_scf_loop()
                if callable(for_channel)
                else state_for_new_potential(channel.eigensolver_state)
                for index, channel in enumerate(channels)
            ),
        )
    restart = getattr(solver, "start_new_scf_loop", None)
    state = (
        restart()
        if callable(restart)
        else state_for_new_potential(previous_result.eigensolver_state)
    )
    return SCFWarmStart(
        input_potential=(
            np.asarray(previous_result.next_effective_potential)
            - np.asarray(previous_system.ionic_potential)
            + np.asarray(system.ionic_potential)
        ),
        hartree_potential=np.asarray(previous_result.hartree_potential),
        eigensolver_state=state,
    )


def compute_forces(system, result, *, symmetrize: bool = True, operations=None):
    """Atomic forces of a finished SCF loop (:mod:`.Forces`)."""

    from .Forces import compute_forces as accelerated_forces

    return accelerated_forces(
        system, result, symmetrize=symmetrize, operations=operations
    )


def ionic_backend(potential_extrapolation: str = "none"):
    """The accelerated operations of an ionic loop (:mod:`parsec_python.Relaxation`).

    ``potential_extrapolation`` is how the SCF loop of moved atoms starts
    (``Potential_Extrapolation``): ``none`` as PARSEC, ``atomic`` with the
    superposed atomic densities moved along, ``second_order`` further along
    the trajectory.  Each backend follows one trajectory.
    """

    from functools import partial

    from parsec_python.Relaxation.ionic_loop import IonicBackend
    from parsec_python.SCF.single_point import extrapolation_keywords

    keywords = extrapolation_keywords(potential_extrapolation)
    return IonicBackend(
        run_scf=run_scf,
        compute_forces=compute_forces,
        update_geometry=update_geometry,
        warm_start=partial(warm_start_for_moved_atoms, **keywords),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


def run_single_point(
    problem: SinglePointInput,
    *,
    backend: BackendName | str = "auto",
    symmetry: SymmetryMode | str = "auto",
    callback: Callable[[SCFIteration], None] | None = None,
) -> AcceleratedSinglePointResult:
    """Prepare and run one accelerated isolated single-point calculation."""

    system = prepare_single_point(
        problem, backend=backend, symmetry=symmetry
    )
    return run_scf(system, callback=callback)


__all__ = [
    "AcceleratedPreparedSinglePointSystem",
    "compute_forces",
    "ionic_backend",
    "prepare_single_point",
    "update_geometry",
    "warm_start_for_moved_atoms",
    "profile_hamiltonian_components",
    "run_scf",
    "run_single_point",
]
