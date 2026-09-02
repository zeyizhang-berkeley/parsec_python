"""Stateful CuPy eigensolver over real reflection representations.

PARSEC assigns an independent eigenproblem to every Abelian representation,
then globally sorts their Ritz values for occupations.  This adapter applies
the same decomposition while reusing the existing CuPy CHEBFF/CHEBDAV and
later-SUBSPACE implementations for each reduced Hamiltonian.

Each representation is assigned round-robin to the CUDA devices selected by
``PARSEC_CUPY_DEVICES``.  Independent sectors run concurrently across devices
and are gathered in fixed PARSEC representation order before the stable global
eigenvalue sort.  A single GPU remains serialized by default because each
sparse/filter kernel already saturates it; ``PARSEC_CUPY_SECTOR_SCHEDULER``
can explicitly request stream overlap without changing the mathematics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.V_ion import NonlocalProjectorOperator

from ..Symmetry import ReflectionRepresentationDecomposition
from ..Symmetry.operator_cache import load_or_build_reduced_operators
from ..backends.cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    cupy_device_count,
    require_cupy,
    synchronize,
)
from .eigval import CuPyEigvalResult, CuPyEigvalSolver


@dataclass(frozen=True)
class CuPySymmetryEigvalState:
    """Per-representation device states retained across SCF iterations."""

    requested_states: int
    sector_state_counts: tuple[int, ...]
    sector_states: tuple[object, ...]
    solves_completed: int


@dataclass(frozen=True)
class CuPySymmetryEigvalResult:
    """Global lowest eigenpairs expanded from their representation wedges."""

    eigenvalues: np.ndarray
    vectors: Any
    residual_norms: np.ndarray | None
    state: CuPySymmetryEigvalState
    solver_path: str
    restarted: bool
    restart_reason: str | None
    representations: np.ndarray
    representation_columns: np.ndarray


@dataclass(frozen=True)
class CuPySymmetryOrbitals:
    """Selected representation vectors retained on the normalized wedge.

    ``scaled_wedge_vectors`` has one row per scalar-field orbit and already
    includes ``1/sqrt(|O_w|)`` on the orbits admitted by each selected
    representation.  Rejected stabilizer orbits are exactly zero.  Its
    squared rows are therefore the physical density on every point in an
    orbit; phases are needed only when materializing signed full-grid states.
    """

    scaled_wedge_vectors: Any
    representations: np.ndarray
    full_to_wedge: np.ndarray
    device_full_to_wedge: Any
    phases: Any
    full_size: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.full_size, int(self.scaled_wedge_vectors.shape[1])

    @property
    def ndim(self) -> int:
        return 2

    def to_full_device(self):
        """Expand signed orbitals once, preserving global eigenvalue order."""

        cp, _ = require_cupy()
        full = cp.empty(self.shape, dtype=cp.float64, order="F")
        for representation in range(int(self.phases.shape[0])):
            output_columns = np.flatnonzero(
                self.representations == representation
            )
            if output_columns.size == 0:
                continue
            expanded = self.scaled_wedge_vectors[
                :, output_columns
            ][self.device_full_to_wedge, :]
            expanded *= self.phases[representation, :, None]
            full[:, output_columns] = expanded
        return full


class CuPySymmetrySCFEigensolver:
    """Merge persistent wedge eigensolvers into one SCF eigensolver callable."""

    def __init__(
        self,
        full_operator: Any,
        negative_laplacian: Any,
        nonlocal_operator: NonlocalProjectorOperator,
        decomposition: ReflectionRepresentationDecomposition,
        *,
        timing_stats: CuPyTimingStats,
        local_potential_getter: Callable[[], np.ndarray] | None = None,
        operator_cache_directory: Path | None = None,
        kinetic_cache_key: str | None = None,
        decomposition_cache_key: str | None = None,
    ) -> None:
        self.full_operator = full_operator
        self.decomposition = decomposition
        self.timing_stats = timing_stats
        self._local_potential_getter = local_potential_getter
        self._state: CuPySymmetryEigvalState | None = None
        self._sector_counts: list[int] | None = None
        self._solvers: list[CuPyEigvalSolver] = []
        self._operators: list[CuPyHamiltonian] = []
        self._sector_timing_stats: list[CuPyTimingStats] = []
        self.scheduler_batches = 0
        self.scheduler_wall_seconds = 0.0

        cp, _ = require_cupy()
        self._primary_device_id = int(cp.cuda.Device().id)
        available_devices = cupy_device_count()
        device_setting = os.environ.get("PARSEC_CUPY_DEVICES", "auto").strip().lower()
        if device_setting in {"", "auto"}:
            device_ids = tuple(range(available_devices))
        elif device_setting in {"current", "off"}:
            device_ids = (self._primary_device_id,)
        else:
            try:
                device_ids = tuple(
                    dict.fromkeys(
                        int(value.strip())
                        for value in device_setting.split(",")
                        if value.strip()
                    )
                )
            except ValueError as error:
                raise ValueError(
                    "PARSEC_CUPY_DEVICES must be 'auto', 'current', or a "
                    "comma-separated list of CUDA device indices"
                ) from error
            if not device_ids or any(
                value < 0 or value >= available_devices for value in device_ids
            ):
                raise ValueError("PARSEC_CUPY_DEVICES contains an unavailable device")
        self.device_ids = device_ids
        self._sector_device_ids = tuple(
            device_ids[index % len(device_ids)]
            for index in range(decomposition.representation_count)
        )
        scheduler = os.environ.get(
            "PARSEC_CUPY_SECTOR_SCHEDULER", "sequential"
        ).strip().lower()
        if scheduler not in {"streams", "sequential"}:
            raise ValueError(
                "PARSEC_CUPY_SECTOR_SCHEDULER must be 'streams' or 'sequential'"
            )
        requested_streams = int(
            os.environ.get(
                "PARSEC_CUPY_SECTOR_STREAMS",
                str(decomposition.representation_count),
            )
        )
        if requested_streams < 1:
            raise ValueError("PARSEC_CUPY_SECTOR_STREAMS must be positive")
        self.scheduler_workers = (
            min(len(device_ids), decomposition.representation_count)
            if len(device_ids) > 1 and scheduler == "sequential"
            else 1
            if scheduler == "sequential"
            else min(requested_streams, decomposition.representation_count)
        )
        self.scheduler_mode = (
            "sequential"
            if self.scheduler_workers == 1
            else "multi-gpu"
            if len(device_ids) > 1
            else "cuda-streams"
        )
        self._streams = []
        for device_id in self._sector_device_ids:
            with cp.cuda.Device(device_id):
                self._streams.append(
                    cp.cuda.Stream.null
                    if self.scheduler_workers == 1
                    else cp.cuda.Stream(non_blocking=True)
                )
        self._executor = (
            None
            if self.scheduler_workers == 1
            else ThreadPoolExecutor(
                max_workers=self.scheduler_workers,
                thread_name_prefix="parsec-cuda-sector",
            )
        )
        collective_lanczos_requested = os.environ.get(
            "PARSEC_CUPY_COLLECTIVE_LANCZOS", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        # Narrow one-vector Lanczos kernels underfill the GPU.  Overlap only
        # those independent representation bounds; the large Chebyshev and
        # orthogonalization phases retain the measured-fast sequential policy.
        self.collective_lanczos = bool(
            collective_lanczos_requested
            and self.scheduler_workers == 1
            and decomposition.representation_count > 1
        )
        self._bound_streams = (
            [cp.cuda.Stream.null] * decomposition.representation_count
            if not self.collective_lanczos
            else [
                cp.cuda.Stream(non_blocking=True)
                for _ in range(decomposition.representation_count)
            ]
        )
        self._bound_executor = (
            None
            if not self.collective_lanczos
            else ThreadPoolExecutor(
                max_workers=decomposition.representation_count,
                thread_name_prefix="parsec-cuda-lanczos",
            )
        )
        with cp.cuda.Device(self._primary_device_id):
            self._device_full_to_wedge = cp.asarray(
                decomposition.reduction.full_to_wedge, dtype=cp.int64
            )
            self._device_phases = cp.asarray(
                decomposition.phases, dtype=cp.float64
            )
            self._device_sector_orbits = tuple(
                cp.asarray(
                    decomposition.sector_orbit_indices(representation),
                    dtype=cp.int64,
                )
                for representation in range(decomposition.representation_count)
            )
            self._device_sector_scales = tuple(
                cp.asarray(
                    1.0
                    / np.sqrt(
                        decomposition.reduction.multiplicities[
                            decomposition.sector_orbit_indices(representation)
                        ]
                    ),
                    dtype=cp.float64,
                )
                for representation in range(decomposition.representation_count)
            )
        self._sector_orbits = tuple(
            decomposition.sector_orbit_indices(representation)
            for representation in range(decomposition.representation_count)
        )

        static_operators = load_or_build_reduced_operators(
            decomposition,
            negative_laplacian,
            nonlocal_operator,
            cache_directory=operator_cache_directory,
            kinetic_key_seed=kinetic_cache_key,
            decomposition_key_seed=decomposition_cache_key,
        )
        self.operator_cache_info = static_operators.cache_info
        symmetric_rows = np.flatnonzero(
            np.all(decomposition.characters == 1, axis=1)
        )
        if symmetric_rows.size != 1:
            raise RuntimeError(
                "reflection character table has no unique totally symmetric sector"
            )
        self.totally_symmetric_representation = int(symmetric_rows[0])
        self.totally_symmetric_negative_laplacian = (
            static_operators.stencil_metadata[
                self.totally_symmetric_representation
            ].to_csr()
        )
        first_neighbors = static_operators.stencil_metadata[0].neighbors
        neighbors_are_shared = all(
            np.array_equal(first_neighbors, item.neighbors)
            for item in static_operators.stencil_metadata[1:]
        )
        shared_device_neighbors: dict[int, Any] = {}
        shared_device_potentials: dict[tuple[int, bytes], Any] = {}
        sector_local_potentials: list[Any] = []
        for representation, (stencil_metadata, reduced_nonlocal) in enumerate(zip(
            static_operators.stencil_metadata,
            static_operators.nonlocal_operators,
            strict=True,
        )):
            device_id = self._sector_device_ids[representation]
            sector_orbits = self._sector_orbits[representation]
            potential_key = (device_id, sector_orbits.tobytes())
            zero_local = np.zeros(
                decomposition.sector_size(representation), dtype=np.float64
            )
            with cp.cuda.Device(device_id):
                sector_timing = CuPyTimingStats()
                operator = CuPyHamiltonian(
                    None,
                    zero_local,
                    reduced_nonlocal,
                    timing_stats=sector_timing,
                    retain_generic_laplacian=False,
                    finite_difference_metadata=stencil_metadata,
                    shared_stencil_neighbors=(
                        shared_device_neighbors.get(device_id)
                        if neighbors_are_shared
                        else None
                    ),
                    shared_effective_potential=shared_device_potentials.get(
                        potential_key
                    ),
                )
                if neighbors_are_shared and device_id not in shared_device_neighbors:
                    selected_stencil = operator.compact_finite_difference
                    shared_device_neighbors[device_id] = getattr(
                        selected_stencil, "neighbors", None
                    )
                if potential_key not in shared_device_potentials:
                    shared_device_potentials[potential_key] = (
                        operator.effective_potential
                    )
                sector_local_potentials.append(operator.effective_potential)
                self._sector_timing_stats.append(sector_timing)
                self._operators.append(operator)
                self._solvers.append(
                    CuPyEigvalSolver(
                        operator,
                        settings=EigvalSettings(safety_buffer=0),
                        timing_stats=sector_timing,
                        retain_vectors_on_device=True,
                        compute_subspace_residuals=False,
                    )
                )
        self._sector_local_potentials = tuple(sector_local_potentials)
        local_potential_buffer_count = len(
            {
                (device_id, int(potential.data.ptr))
                for device_id, potential in zip(
                    self._sector_device_ids,
                    self._sector_local_potentials,
                    strict=True,
                )
            }
        )
        device_count = len(set(self._sector_device_ids))
        if local_potential_buffer_count == device_count:
            self.local_potential_storage = (
                "one persistent shared device buffer per CUDA device"
            )
        else:
            self.local_potential_storage = (
                "one persistent device buffer per distinct "
                "stabilizer-filtered sector map and CUDA device"
            )
        # Static uploads occurred sequentially during construction and are
        # therefore additive wall time.  Later per-sector counters are merged
        # only after worker completion, avoiding shared Python mutations.
        self.timing_stats.initialization_seconds += sum(
            item.initialization_seconds for item in self._sector_timing_stats
        )
        storage_modes = {
            (
                operator.compact_finite_difference.storage_mode
                if operator.compact_finite_difference is not None
                else "float64_csr"
            )
            for operator in self._operators
        }
        self.finite_difference_storage = (
            storage_modes.pop() if len(storage_modes) == 1 else "mixed"
        )
        self.finite_difference_neighbors = (
            "shared_across_representations"
            if neighbors_are_shared and shared_device_neighbors is not None
            else "private_per_representation"
        )

    @property
    def state(self) -> CuPySymmetryEigvalState | None:
        return self._state

    @property
    def representation_count(self) -> int:
        return self.decomposition.representation_count

    @property
    def fused_projector_scatter(self) -> bool:
        """Whether every nonempty sector fuses the large KB scatter."""

        relevant = [operator for operator in self._operators if operator.projector_count]
        return bool(relevant) and all(
            operator.fused_projector_scatter for operator in relevant
        )

    @property
    def custom_projector_projection(self) -> bool:
        """Whether every nonempty sector replaces tiny cuSPARSE B.T calls."""

        relevant = [operator for operator in self._operators if operator.projector_count]
        return bool(relevant) and all(
            operator.custom_projector_projection is not None
            for operator in relevant
        )

    @property
    def projector_reduction_modes(self) -> str:
        """Report the selected short/long-row CUDA dot policies."""

        modes = {
            getattr(operator.custom_projector_projection, "reduction_mode", "none")
            for operator in self._operators
            if operator.projector_count
        }
        return "+".join(sorted(modes)) if modes else "none"

    @property
    def mixed_precision_filter(self) -> bool:
        """Whether every sector uses FP32 only inside later filtering."""

        return bool(self._operators) and all(
            operator.mixed_precision_recurrence is not None
            for operator in self._operators
        )

    def reset(self) -> None:
        """Discard every saved representation subspace."""

        for solver in self._solvers:
            solver.reset()
        self._state = None
        self._sector_counts = None

    def _initial_sector_count(
        self,
        representation: int,
        requested_states: int,
        safety_buffer: int,
    ) -> int:
        # PARSEC initeigval: nadd + nstate/nrep (integer division).  The ceil
        # guard matters only for an unusual user choice nadd=0 and ensures the
        # union contains at least the globally requested number of states.
        count = max(
            (requested_states + self.representation_count - 1)
            // self.representation_count,
            requested_states // self.representation_count + safety_buffer,
        )
        return min(self.decomposition.sector_size(representation), max(1, count))

    def _update_local_potential(self, full_potential) -> None:
        """Upload one shared invariant wedge field for all sector operators."""

        cp, _ = require_cupy()
        from ..SCF.symmetry_fields import SymmetryScalarField

        if isinstance(full_potential, SymmetryScalarField):
            if full_potential.reduction is not self.decomposition.reduction:
                raise ValueError("local potential uses a different symmetry map")
            wedge = full_potential.values
        else:
            wedge = self.decomposition.invariant_wedge_values(full_potential)
        started = perf_counter()
        host_wedge = np.ascontiguousarray(wedge, dtype=np.float64)
        updated_buffers: set[int] = set()
        for representation, (device_id, device) in enumerate(
            zip(
                self._sector_device_ids,
                self._sector_local_potentials,
                strict=True,
            )
        ):
            pointer = int(device.data.ptr)
            if pointer in updated_buffers:
                continue
            with cp.cuda.Device(device_id):
                device.set(host_wedge[self._sector_orbits[representation]])
            updated_buffers.add(pointer)
        # Sector Hamiltonians share the FP64 field above.  Their optional
        # mixed-precision Chebyshev recurrences own private FP32 shadows, so
        # refresh those shadows from the just-updated device field before any
        # sector is scheduled.  No host transfer is repeated here.
        for device_id, operator in zip(
            self._sector_device_ids, self._operators, strict=True
        ):
            recurrence = operator.mixed_precision_recurrence
            if recurrence is not None:
                with cp.cuda.Device(device_id):
                    recurrence.update_potential(
                        operator.effective_potential
                    )
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                synchronize()
        self.timing_stats.potential_update_seconds += perf_counter() - started

    @staticmethod
    def _global_order(
        results: list[CuPyEigvalResult],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.concatenate([result.eigenvalues for result in results])
        representations = np.concatenate(
            [
                np.full(result.eigenvalues.size, index, dtype=np.int32)
                for index, result in enumerate(results)
            ]
        )
        columns = np.concatenate(
            [
                np.arange(result.eigenvalues.size, dtype=np.int32)
                for result in results
            ]
        )
        order = np.argsort(values, kind="stable")
        return values[order], representations[order], columns[order]

    def _solve_sectors(
        self,
        settings: EigvalSettings,
        counts: list[int],
        previous_state_supplied: bool,
    ) -> tuple[list[CuPyEigvalResult], bool, list[str]]:
        sector_settings = replace(settings, safety_buffer=0)
        results: list[CuPyEigvalResult] = []
        restarted = False
        reasons: list[str] = []
        solved = self._run_sector_jobs(
            tuple(range(self.representation_count)),
            counts,
            sector_settings,
            reset=not previous_state_supplied,
        )
        for representation in range(self.representation_count):
            result = solved[representation]
            results.append(result)
            restarted = restarted or result.restarted
            if result.restart_reason is not None:
                reasons.append(
                    f"representation {representation + 1}: {result.restart_reason}"
                )
        return results, restarted, reasons

    def _run_one_sector(
        self,
        representation: int,
        count: int,
        settings: EigvalSettings,
        *,
        reset: bool,
        spectral_bound: Any | None = None,
    ) -> CuPyEigvalResult:
        """Run one independent solver on its assigned CUDA stream."""

        cp, _ = require_cupy()
        with cp.cuda.Device(self._sector_device_ids[representation]):
            with self._streams[representation]:
                solver = self._solvers[representation]
                if reset:
                    solver.reset()
                return solver.solve(
                    count,
                    settings=settings,
                    spectral_bound=spectral_bound,
                )

    def _run_one_bound(
        self,
        representation: int,
        count: int,
        settings: EigvalSettings,
        *,
        reset: bool,
    ):
        """Prepare one sector's unchanged PARSEC Lanczos bound on its stream."""

        cp, _ = require_cupy()
        with cp.cuda.Device(self._sector_device_ids[representation]):
            with self._bound_streams[representation]:
                return self._solvers[representation].prepare_spectral_bound(
                    count,
                    settings=settings,
                    reset=reset,
                )

    def _run_sector_jobs(
        self,
        representations: tuple[int, ...],
        counts: list[int],
        settings: EigvalSettings,
        *,
        reset: bool,
    ) -> dict[int, CuPyEigvalResult]:
        """Submit independent sectors and merge timing after synchronization."""

        before = [item.as_dict() for item in self._sector_timing_stats]
        started = perf_counter()
        spectral_bounds: dict[int, Any] = {}
        if self._bound_executor is not None and len(representations) > 1:
            bound_futures = {
                representation: self._bound_executor.submit(
                    self._run_one_bound,
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                )
                for representation in representations
            }
            spectral_bounds = {
                representation: bound_futures[representation].result()
                for representation in representations
            }
        if self._executor is None:
            results = {
                representation: self._run_one_sector(
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                    spectral_bound=spectral_bounds.get(representation),
                )
                for representation in representations
            }
        else:
            futures = {
                representation: self._executor.submit(
                    self._run_one_sector,
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                    spectral_bound=spectral_bounds.get(representation),
                )
                for representation in representations
            }
            # Dictionary insertion and retrieval follow the supplied PARSEC
            # representation order, independent of task completion order.
            results = {
                representation: futures[representation].result()
                for representation in representations
            }
        self.scheduler_batches += 1
        self.scheduler_wall_seconds += perf_counter() - started

        for sector_stats, snapshot in zip(
            self._sector_timing_stats, before, strict=True
        ):
            current = sector_stats.as_dict()
            for name, old_value in snapshot.items():
                if name == "initialization_seconds":
                    continue
                delta = current[name] - old_value
                setattr(
                    self.timing_stats,
                    name,
                    getattr(self.timing_stats, name) + delta,
                )
        return results

    def _ensure_spectral_bracket(
        self,
        results: list[CuPyEigvalResult],
        counts: list[int],
        requested_states: int,
        settings: EigvalSettings,
    ) -> tuple[list[CuPyEigvalResult], bool, list[str]]:
        """Grow sectors whose last computed value does not bracket the cutoff."""

        restarted = False
        reasons: list[str] = []
        for _ in range(8):
            values, representations, _ = self._global_order(results)
            if values.size < requested_states:
                raise RuntimeError("symmetry sectors returned too few eigenvalues")
            cutoff = float(values[requested_states - 1])
            grow: list[int] = []
            for representation, result in enumerate(results):
                sector_size = self.decomposition.sector_size(representation)
                if counts[representation] >= sector_size:
                    continue
                scale = max(1.0, abs(cutoff), abs(float(result.eigenvalues[-1])))
                if float(result.eigenvalues[-1]) <= cutoff + 1.0e-11 * scale:
                    grow.append(representation)
            if not grow:
                return results, restarted, reasons

            increment = max(1, settings.safety_buffer)
            sector_settings = replace(settings, safety_buffer=0)
            for representation in grow:
                old_count = counts[representation]
                counts[representation] = min(
                    self.decomposition.sector_size(representation),
                    max(old_count + increment, (3 * old_count + 1) // 2),
                )
                restarted = True
                reasons.append(
                    f"representation {representation + 1}: spectral bracket "
                    f"grew {old_count}->{counts[representation]}"
                )
            grown = self._run_sector_jobs(
                tuple(grow), counts, sector_settings, reset=False
            )
            for representation in grow:
                results[representation] = grown[representation]
        raise RuntimeError(
            "symmetry representation state allocation did not bracket the "
            "global requested eigenspectrum"
        )

    def _trim_sector_states_like_parsec(
        self,
        results: list[CuPyEigvalResult],
        counts: list[int],
        requested_states: int,
        safety_buffer: int,
    ) -> None:
        """Apply ``eigen_sort``'s active-state count for the next SCF step.

        PARSEC first counts each representation among the lowest
        ``N_states-1`` globally sorted values, then admits at most ``nadd``
        additional values per representation from the remainder.  Its
        eigenspace allocation is not shrunk, but ``nn`` limits later work.
        Here a device view of the leading Ritz columns is the equivalent.
        """

        if safety_buffer < 1:
            return
        _, representations, _ = self._global_order(results)
        base = np.bincount(
            representations[: max(0, requested_states - 1)],
            minlength=self.representation_count,
        )
        extra = np.zeros(self.representation_count, dtype=np.int64)
        for representation in representations[max(0, requested_states - 1) :]:
            index = int(representation)
            if extra[index] < safety_buffer:
                extra[index] += 1
            if np.all(extra >= safety_buffer):
                break
        desired = np.maximum(1, base + extra)
        for representation, solver in enumerate(self._solvers):
            new_count = min(counts[representation], int(desired[representation]))
            if new_count < counts[representation]:
                # truncate_state creates only leading-column views; it does
                # not launch a kernel and is therefore device-context neutral.
                solver.truncate_state(new_count)
                counts[representation] = new_count

    def _pack_selected_wedge_vectors(
        self,
        results: list[CuPyEigvalResult],
        selected_representations: np.ndarray,
        selected_columns: np.ndarray,
    ) -> CuPySymmetryOrbitals:
        """Pack globally sorted orbitals without repeated full-grid expansion."""

        cp, _ = require_cupy()
        count = int(selected_representations.size)
        with cp.cuda.Device(self._primary_device_id):
            wedge = cp.zeros(
                (self.decomposition.wedge_size, count),
                dtype=cp.float64,
                order="F",
            )
        for representation in range(self.representation_count):
            output_columns = np.flatnonzero(
                selected_representations == representation
            )
            if output_columns.size == 0:
                continue
            wedge_columns = selected_columns[output_columns]
            source_device = self._sector_device_ids[representation]
            with cp.cuda.Device(source_device):
                source = results[representation].vectors[:, wedge_columns]
                # Materialize the advanced-indexing result before changing
                # device contexts.  Buffered sector subspaces never move.
                source = cp.asfortranarray(source)
            with cp.cuda.Device(self._primary_device_id):
                if source_device == self._primary_device_id:
                    primary_source = source
                else:
                    try:
                        # CuPy uses CUDA peer access when the device pair
                        # permits it.  Materialize a primary-device copy before
                        # the stabilizer-aware scatter below.
                        primary_source = cp.asarray(source)
                    except Exception:
                        # Exact host-staged fallback for devices without P2P.
                        with cp.cuda.Device(source_device):
                            host_source = source.get()
                        with cp.cuda.Device(self._primary_device_id):
                            primary_source = cp.asarray(host_source)
                scaled = primary_source * self._device_sector_scales[
                    representation
                ][:, None]
                sector_orbits = self._device_sector_orbits[representation]
                wedge[
                    sector_orbits[:, None],
                    cp.asarray(output_columns, dtype=cp.int64)[None, :],
                ] = scaled
        return CuPySymmetryOrbitals(
            scaled_wedge_vectors=wedge,
            representations=np.ascontiguousarray(
                selected_representations, dtype=np.int32
            ),
            full_to_wedge=self.decomposition.reduction.full_to_wedge,
            device_full_to_wedge=self._device_full_to_wedge,
            phases=self._device_phases,
            full_size=self.decomposition.full_size,
        )

    def __call__(
        self,
        operator: Any,
        requested_states: int,
        *,
        settings: EigvalSettings,
        state: object | None = None,
    ) -> CuPySymmetryEigvalResult:
        """Solve all sectors, globally sort them, and expand selected states."""

        if operator is not self.full_operator:
            raise ValueError("symmetry SCF received a different full Hamiltonian")
        if state is None:
            self.reset()
        elif state is not self._state:
            raise ValueError("SCF state does not belong to this symmetry eigensolver")

        requested_states = int(requested_states)
        if requested_states < 1:
            raise ValueError("requested_states must be positive")
        if self._sector_counts is None:
            self._sector_counts = [
                self._initial_sector_count(
                    representation,
                    requested_states,
                    settings.safety_buffer,
                )
                for representation in range(self.representation_count)
            ]
        counts = self._sector_counts

        # CuPyHamiltonianBackend.bind has already retained this exact host
        # field on the full backend.  It is invariant by construction after
        # symmetry-sector densities are used; orbit averaging removes only
        # roundoff before the wedge upload.
        full_potential = (
            np.asarray(self.full_operator.effective_potential.get(), dtype=np.float64)
            if self._local_potential_getter is None
            else self._local_potential_getter()
        )
        self._update_local_potential(full_potential)

        previous = state is not None
        results, restarted, reasons = self._solve_sectors(
            settings, counts, previous
        )
        results, bracket_restarted, bracket_reasons = self._ensure_spectral_bracket(
            results, counts, requested_states, settings
        )
        restarted = restarted or bracket_restarted
        reasons.extend(bracket_reasons)

        values, representations, columns = self._global_order(results)
        selected_values = np.asarray(values[:requested_states], dtype=np.float64)
        selected_representations = representations[:requested_states]
        selected_columns = columns[:requested_states]
        vectors = self._pack_selected_wedge_vectors(
            results, selected_representations, selected_columns
        )

        residual_norms = None
        if all(result.residual_norms is not None for result in results):
            residual_norms = np.asarray(
                [
                    results[int(rep)].residual_norms[int(column)]
                    for rep, column in zip(
                        selected_representations,
                        selected_columns,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )

        self._trim_sector_states_like_parsec(
            results,
            counts,
            requested_states,
            settings.safety_buffer,
        )

        solves_completed = 1 if self._state is None else self._state.solves_completed + 1
        self._state = CuPySymmetryEigvalState(
            requested_states=requested_states,
            sector_state_counts=tuple(counts),
            sector_states=tuple(
                solver.device_state for solver in self._solvers
            ),
            solves_completed=solves_completed,
        )
        paths = {result.solver_path for result in results}
        solver_path = paths.pop() if len(paths) == 1 else "mixed"
        return CuPySymmetryEigvalResult(
            eigenvalues=selected_values,
            vectors=vectors,
            residual_norms=residual_norms,
            state=self._state,
            solver_path=solver_path,
            restarted=restarted,
            restart_reason="; ".join(reasons) if reasons else None,
            representations=np.asarray(selected_representations, dtype=np.int32),
            representation_columns=np.asarray(selected_columns, dtype=np.int32),
        )


__all__ = [
    "CuPySymmetryEigvalResult",
    "CuPySymmetryEigvalState",
    "CuPySymmetryOrbitals",
    "CuPySymmetrySCFEigensolver",
]
