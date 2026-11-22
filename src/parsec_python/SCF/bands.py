"""Band structure along a k-path after the SCF (Fortran ``calc_bands``).

Like Fortran PARSEC, the band structure is a non-self-consistent
diagonalization in the converged self-consistent potential:

* the path is a list of segments in relative (reciprocal-lattice) coordinates;
  the shortest segment gets ``bandstruc_points`` k-points and the others
  proportionally more (:func:`~parsec_python.Grid.kpoints.band_path_kpoints`);
* at every k-point the Bloch Hamiltonian (kinetic with ``-2i k.grad``,
  Bloch-phase nonlocal and spin-orbit projectors) is built from the SCF
  potential and diagonalized from scratch by the SCF's *first* eigensolver,
  CHEBFF (``solver%eig_init = .false.`` in ``calc_bands``), i.e.
  ``FF_MaxIter`` Chebyshev filter cycles of degree ``Chebdav_Degree``.  PARSEC's
  CHEBFF has no convergence test there (its ``Diag_Tolerance`` default -1 for
  CHEBFF); here ``PARSEC_BAND_CONVERGENCE=auto`` (the default since 2026-11-21,
  :func:`band_convergence_setting`) continues the cycles to CHEBDAV's residual
  test, and ``off`` keeps PARSEC's fixed cycles.  The largest
  residual ``|H v - e v|`` is reported for every k-point as a diagnostic.

Extension beyond Fortran: CHEBDAV adds ``Matvec_Blocksize`` vectors per Davidson
step, and at a highly symmetric k-point whose degenerate multiplet is larger than
the block (an 8-fold level in a folded cubic cell, 16-fold with spin-orbit) it can
leave reported bands unconverged, at a block size that is not predictable.  When a
reported band's residual exceeds ``residual_tolerance`` the k-point is re-solved
with a larger block and a new random start (up to ``max_attempts`` in total) and
the attempt with the smallest residual is kept; the number of attempts per k-point
is recorded in ``BandStructureResult.attempts``.  A device worker that runs the
device-resident first solve (the default on a GPU, see :func:`run_band_structure`)
re-solves the same way from a new random start; it has no Davidson block.

The Hamiltonian follows the SCF mode, so the same ``bandstruc`` block gives
scalar, collinear spin-polarized (up/down), spin-orbit, spin-orbit +
collinear-spin and non-collinear bands.  Comparing runs with and without
``SO_PSP``/``SO_from_scratch`` gives the effect of spin-orbit coupling.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..Eigensolvers import solve_eigval
from ..Grid import band_path_kpoints, explicit_kpoints
from ..Laplacian.finite_difference import bloch_laplacian_or_none
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..models import BandStructureResult, BandStructureSettings
from .kpoint_solver import device_solver_supports, map_tasks
from .kpoints_soc import _eigval_settings, _k_point_operators, _spinor_hamiltonian

MODES = ("scalar", "spin_polarized", "soc", "soc_spin_polarized", "noncollinear")

BAND_CONVERGENCE_SWITCH = "PARSEC_BAND_CONVERGENCE"
BAND_REPORTED_RESIDUALS_SWITCH = "PARSEC_BAND_REPORTED_RESIDUALS"


def band_convergence_setting() -> str:
    """``PARSEC_BAND_CONVERGENCE``: ``auto`` (the default since 2026-11-21) makes the CHEBFF first solve of every
    band and ``nscf`` k-point (``Eigensolver: chebff``) a converged one: after the ``Bands_FF_MaxIter`` cycles
    (``FF_MaxIter``), CHEBFF cycles go on until the residuals of the reported bands pass CHEBDAV's relative test
    with the input's ``Diag_Tolerance`` (times their largest ``|e|``; ``1e-4`` where the input gives none or
    PARSEC's CHEBFF default ``-1``) or ``PARSEC_KPOINT_EXTRA_CYCLES`` (24) more cycles were made, as the
    device-resident first solve of ``Eigensolver: chebdav`` does
    (:class:`parsec_python.Eigensolvers.eigval.ConvergedFirstSolveSettings`).  A k-point converged after the fixed
    cycles keeps its levels bit for bit.  ``off``: the fixed cycles, PARSEC 1.5's ``calc_bands`` with its CHEBFF
    default ``Diag_Tolerance -1``, as before (1000 diamond atoms: largest residual 0.10 Ry at the 3 cycles)."""

    import os

    value = os.environ.get(BAND_CONVERGENCE_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{BAND_CONVERGENCE_SWITCH} must be auto or off, not {value!r}")
    return value


def band_reported_residuals_setting() -> str:
    """``PARSEC_BAND_REPORTED_RESIDUALS``: which states the residual test of a converged first solve covers at a band
    k-point solved in the sectors of its little group.  ``auto``: the levels the k-point reports and the pairs that a
    residual bound leaves possibly among them, a heuristic that does not prove that no level is missing
    (:meth:`parsec_python.acceleration.SCF.kpoint_sectors.SectorKPointSolver.solve_reported`).  A k-point that stays
    whole keeps the test of all its states at ``Diag_Tolerance``.  ``off``: every state of every sector, with the
    margin no k-point reports, as before."""

    import os

    value = os.environ.get(BAND_REPORTED_RESIDUALS_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{BAND_REPORTED_RESIDUALS_SWITCH} must be auto or off, not {value!r}")
    return value


def validate_band_settings() -> None:
    """Read the switches of the first solve of the band and ``nscf`` k-points (``PARSEC_BAND_CONVERGENCE``,
    ``PARSEC_BAND_REPORTED_RESIDUALS``, ``PARSEC_KPOINT_EXTRA_CYCLES``), so that a value they do not know is refused
    before the SCF."""

    from ..Eigensolvers.eigval import first_solve_extra_cycles

    band_convergence_setting()
    band_reported_residuals_setting()
    first_solve_extra_cycles()


def converged_band_settings(eigval_settings, converge: bool | None = None, tolerance: float | None = None):
    """``eigval_settings`` of the band k-points: :class:`ConvergedFirstSolveSettings` for a CHEBFF first solve
    under ``converge`` (``None``: ``PARSEC_BAND_CONVERGENCE=auto``), else unchanged.  ``tolerance`` (the
    input's ``Diag_Tolerance``) is the relative residual its cycles go on to; ``None`` keeps that of
    ``eigval_settings.chebdav``."""

    from ..Eigensolvers.eigval import ConvergedFirstSolveSettings, EigvalSettings

    if converge is None:
        converge = band_convergence_setting() == "auto"
    if not converge or eigval_settings.initial_method != "chebff":
        return eigval_settings
    values = {item.name: getattr(eigval_settings, item.name) for item in dataclasses.fields(EigvalSettings)}
    if tolerance is not None and float(tolerance) > 0.0:
        # the settings of a CHEBFF SCF carry CHEBDAV's defaults: the test takes the tolerance of the input
        values["chebdav"] = dataclasses.replace(values["chebdav"], convergence_tolerance=float(tolerance))
    return ConvergedFirstSolveSettings(**values)


def band_mode(problem) -> str:
    """Hamiltonian family of the SCF described by ``problem``."""
    scf = problem.scf
    has_soc_species = any(
        specification.spin_orbit for specification in problem.pseudopotentials.values()
    )
    if scf.noncollinear:
        return "noncollinear"
    if scf.self_consistent_spin_orbit:
        return "soc_spin_polarized" if scf.spin_polarized else "soc"
    if scf.spin_polarized:
        if has_soc_species:
            raise ValueError(
                "bands for the perturbative spin-orbit route (Spin_Polarization "
                "with SO_PSP but no SO_from_scratch) are not supported; use "
                "SO_from_scratch for spin-orbit bands, or SO_PSP: .false. for "
                "scalar-relativistic spin-polarized bands"
            )
        return "spin_polarized"
    return "scalar"


def band_potentials(result, mode: str) -> dict[str, np.ndarray]:
    """The converged local potentials a band calculation needs, from an SCF result.

    With LDA+U the result's potential matrices ``D`` (``result.hubbard``,
    shape ``(channels, atoms, 5, 5)``) come along under ``"hubbard"``: the
    bands are those of the converged Hamiltonian, LDA+U term included, with
    the Bloch projectors of every band k-point (``calc_bands.f90`` keeps the
    projectors of the SCF k-points instead, a Fortran defect).
    """
    hubbard = getattr(result, "hubbard", None)
    extra = {} if hubbard is None else {"hubbard": np.asarray(hubbard.potentials)}
    if mode == "scalar":
        return {"effective_potential": np.asarray(result.output_effective_potential), **extra}
    potentials = getattr(result, "scf_potentials", None)
    if not potentials:
        raise ValueError(f"the SCF result carries no converged potentials for mode {mode!r}")
    return {**{name: np.asarray(value) for name, value in potentials.items()}, **extra}


_REQUIRED = {
    "scalar": ("effective_potential",),
    "spin_polarized": ("potential_up", "potential_down"),
    "soc": ("effective_potential",),
    "soc_spin_polarized": ("effective_potential", "xc_delta"),
    "noncollinear": ("effective_potential", "xc_field"),
}


def band_path_cartesian(
    settings: BandStructureSettings, cell
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Cartesian k-points (1/bohr) of every segment and the segment names."""
    starts = np.asarray([segment.start for segment in settings.segments])
    ends = np.asarray([segment.end for segment in settings.segments])
    start_cartesian, _ = explicit_kpoints(
        starts, unit="reciprocal_lattice_vectors", cell=cell
    )
    end_cartesian, _ = explicit_kpoints(ends, unit="reciprocal_lattice_vectors", cell=cell)
    segments = band_path_kpoints(
        start_cartesian, end_cartesian, settings.points, rounding=getattr(settings, "rounding", "truncate")
    )
    return segments, tuple(segment.name for segment in settings.segments)


def load_scf_for_bands(archive_path, problem, system, mode: str):
    """Converged potentials, Fermi level and state count from an SCF result archive.

    Lets a band structure be recomputed (another path, solver settings, ...) without
    repeating the SCF.  The archive must come from the same input: the grid size and
    atom positions are checked.
    """
    # Closed when done (an error raised below keeps this frame, and an open archive would keep its
    # file, alive: Windows then cannot remove the folder of the archive).
    with np.load(archive_path) as archive:
        files = set(archive.files)
        if "atom_coordinates_bohr" in files:
            stored = np.asarray(archive["atom_coordinates_bohr"], dtype=float)
            current = np.asarray([atom.position for atom in system.atoms], dtype=float)
            if stored.shape != current.shape or not np.allclose(stored, current, atol=1e-6):
                raise ValueError(f"{archive_path} was written for different atom positions")
        if mode == "scalar":
            keys = {"effective_potential": "output_effective_potential_ry"}
        else:
            keys = {
                name: f"scf_potential_{name}" for name in _REQUIRED[mode]
            }
        missing = [key for key in keys.values() if key not in files]
        if missing:
            raise ValueError(
                f"{archive_path} lacks {missing}; rerun the SCF with this version "
                f"to store the converged potentials needed for mode {mode!r}"
            )
        potentials = {name: np.asarray(archive[key]) for name, key in keys.items()}
        for name, value in potentials.items():
            if value.shape[-1] != system.grid.size:
                raise ValueError(
                    f"{archive_path}: potential {name!r} has {value.shape[-1]} grid points, "
                    f"the input's grid has {system.grid.size}"
                )
        eigenvalue_key = "eigenvalues_up_ry" if mode == "spin_polarized" else "eigenvalues_ry"
        k_count = int(archive["n_kpoints"]) if "n_kpoints" in files else 1
        number_of_states = int(np.asarray(archive[eigenvalue_key]).size // k_count)
        if "run_states_per_kpoint" in files:
            # the states of a k-point of the run (a mesh solved as its supercell lists fewer levels at some)
            number_of_states = int(archive["run_states_per_kpoint"])
        return potentials, float(archive["fermi_level_ry"]), number_of_states


def spin_expectation(spinors: np.ndarray) -> np.ndarray:
    """``<sigma_x>, <sigma_y>, <sigma_z>`` of every spinor column, shape ``(n, 3)``."""
    n_grid = spinors.shape[0] // 2
    up, down = spinors[:n_grid], spinors[n_grid:]
    cross = np.sum(np.conj(up) * down, axis=0)
    return np.column_stack(
        (
            2.0 * cross.real,
            2.0 * cross.imag,
            np.sum(np.abs(up) ** 2, axis=0) - np.sum(np.abs(down) ** 2, axis=0),
        )
    )


def run_band_structure(
    system,
    settings: BandStructureSettings,
    *,
    mode: str,
    potentials: dict[str, np.ndarray],
    fermi_level: float,
    number_of_states: int,
    callback: Callable[[int, int, np.ndarray, float, int], None] | None = None,
    residual_tolerance: float = 2.0e-2,
    max_attempts: int = 5,
    workers: int | str | None = None,
    backend: str | None = None,
    symmetry=None,
) -> BandStructureResult:
    """Diagonalize the SCF Hamiltonian along ``settings``' path.

    ``potentials`` are the converged local potentials (:func:`band_potentials`),
    ``number_of_states`` the number of bands reported per k-point (and per
    spin channel).  ``callback(k_index, n_kpoints, eigenvalues, residual,
    attempts)`` is called as each k-point finishes (in completion order when
    several workers run).  ``residual_tolerance`` [Ry] and ``max_attempts``
    control the CHEBDAV re-solve described in the module docstring (a CHEBFF
    first solver is never retried).  ``symmetry`` is that of
    :func:`solve_kpoint_bands` (the accelerated Gamma-cell SCF in sectors).

    The k-points of the path are independent, so they are spread over workers
    (serial without the acceleration package).  ``backend`` (or
    ``PARSEC_KPOINT_BACKEND``) selects ``cpu`` (``workers`` worker threads;
    ``None``: ``PARSEC_KPOINT_WORKERS``, default automatic; see
    :mod:`parsec_python.acceleration.Parallel`), ``cupy`` (one worker thread
    per CUDA device of ``PARSEC_CUPY_DEVICES``, whatever ``workers`` says),
    ``auto`` (``cupy`` when CuPy sees a device, else ``cpu``) or
    ``numpy-device`` (the device code on the host, ``PARSEC_KPOINT_WORKERS``
    threads).  Without either, the device of
    ``PARSEC_DEVICE`` (``--device``, default ``auto``) decides: a GPU when CuPy
    sees one, the CPU otherwise.  A device worker runs the Hamiltonian on the
    device and, unless ``PARSEC_KPOINT_SOLVER=host`` (the host eigensolver with
    only the Chebyshev filter on the device), the device-resident first solve
    (CHEBFF, with more CHEBFF cycles to CHEBDAV's residual test for
    ``Eigensolver: chebdav``, and for ``Eigensolver: chebff`` under
    ``PARSEC_BAND_CONVERGENCE=auto``).
    Each k-point is the same computation whatever the worker count or device,
    so the bands do not depend on them (to round-off); the host and the
    device-resident eigensolvers are different algorithms and agree to the
    eigensolver tolerance.
    """
    if getattr(system, "hubbard", None) is not None and "hubbard" not in potentials:
        raise NotImplementedError(
            "a band structure with LDA+U needs the potential matrices of the SCF "
            "(the 'hubbard' entry of band_potentials); an archive does not hold them"
        )

    cell = system.input.periodic_cell
    segments, names = band_path_cartesian(settings, cell)
    k_points = np.concatenate(segments)
    counts = np.asarray([len(segment) for segment in segments])
    bands = solve_kpoint_bands(
        system,
        k_points,
        mode=mode,
        potentials=potentials,
        number_of_states=number_of_states,
        callback=callback,
        residual_tolerance=residual_tolerance,
        max_attempts=max_attempts,
        workers=workers,
        backend=backend,
        filter_cycles=settings.ff_maxiter,
        symmetry=symmetry,
    )
    return BandStructureResult(
        mode=mode,
        channels=bands.channels,
        segment_names=names,
        segment_counts=counts,
        k_points=k_points,
        eigenvalues=bands.eigenvalues,
        residual_norms=bands.residual_norms,
        fermi_level=float(fermi_level),
        spin_expectation=bands.spin_expectation,
        attempts=bands.attempts,
        residual_scope=bands.residual_scope,
    )


@dataclasses.dataclass
class KPointBands:
    """Eigenvalues of the fixed-potential Hamiltonian at a list of k-points.

    ``eigenvalues`` (Ry) and ``residual_norms`` are indexed ``(channel, k)``
    as in :class:`~parsec_python.models.BandStructureResult`;
    ``attempts`` counts the diagonalizations of each ``(channel, k)``.
    """

    channels: tuple[str, ...]
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    attempts: np.ndarray
    spin_expectation: np.ndarray | None = None
    # With ``keep_vectors``: ``vectors[channel][k]`` the ``(n_grid, n_states)``
    # eigenvectors (Bloch factors ``u_nk``) of the reported levels; ``None``
    # otherwise.
    vectors: list | None = None
    # ``"reported levels"`` where the residuals were tested on those levels alone in the sectors of the k-points
    # (PARSEC_BAND_REPORTED_RESIDUALS=auto); ``None`` otherwise.
    residual_scope: str | None = None


def solve_kpoint_bands(
    system,
    k_points: np.ndarray,
    *,
    mode: str,
    potentials: dict[str, np.ndarray],
    number_of_states: int,
    callback: Callable[[int, int, np.ndarray, float, int], None] | None = None,
    residual_tolerance: float = 2.0e-2,
    max_attempts: int = 5,
    workers: int | str | None = None,
    backend: str | None = None,
    keep_vectors: bool = False,
    filter_cycles: int | None = None,
    symmetry=None,
    converge: bool | None = None,
) -> KPointBands:
    """Diagonalize the Hamiltonian of ``potentials`` at every row of ``k_points``.

    The k-point loop of :func:`run_band_structure` (and of the
    non-self-consistent run of :mod:`parsec_python.SCF.nscf`): every k-point
    (Cartesian, 1/bohr) is solved from scratch by the SCF's first
    eigensolver, with the CHEBDAV re-solve and the worker pool described
    there.  ``callback`` is that of :func:`run_band_structure`.
    ``keep_vectors`` keeps the eigenvectors of the reported levels
    (``KPointBands.vectors``; host arrays).  ``filter_cycles`` replaces
    ``FF_MaxIter`` in these solves (PARSEC 1.5's ``Bands_FF_MaxIter``,
    ``calc_bands.f90:336-338``; CHEBFF first solver only); ``None`` keeps it.
    ``symmetry``: the operations the potential is invariant under, given by
    the accelerated SCF of a Gamma cell in the sectors of its point group
    (:class:`parsec_python.acceleration.SCF.band_sectors.BandSymmetry`); a
    scalar path then solves its k-points in the sectors of their little
    groups where they split (``PARSEC_BAND_KPOINT_SECTORS``).  ``None`` (the
    reference CLI) keeps every k-point whole.  ``converge``: whether a CHEBFF
    first solve continues to CHEBDAV's residual test (:func:`converged_band_settings`;
    ``None``: ``PARSEC_BAND_CONVERGENCE``).
    """

    if mode not in MODES:
        raise ValueError(f"unknown band mode {mode!r}; choose from {MODES}")
    missing = [name for name in _REQUIRED[mode] if name not in potentials]
    if missing:
        raise ValueError(f"mode {mode!r} needs the potentials {missing}")
    k_points = np.asarray(k_points, dtype=float)

    eigval_settings = _eigval_settings(system, system.input.eigensolver.filter_degree)
    if filter_cycles is not None:
        eigval_settings = dataclasses.replace(
            eigval_settings,
            chebff=dataclasses.replace(eigval_settings.chebff, filter_cycles=int(filter_cycles)),
        )
    # PARSEC_BAND_CONVERGENCE=auto: a CHEBFF first solve continues to CHEBDAV's residual test with Diag_Tolerance
    eigval_settings = converged_band_settings(
        eigval_settings, converge, tolerance=getattr(system.input.eigensolver, "tolerance", None)
    )
    spinor = mode in ("soc", "soc_spin_polarized", "noncollinear")
    channels = ("up", "down") if mode == "spin_polarized" else ("all",)
    n_k = k_points.shape[0]
    eigenvalues = np.zeros((len(channels), n_k, number_of_states))
    residuals = np.zeros((len(channels), n_k))
    attempts = np.ones((len(channels), n_k), dtype=int)
    sigma = np.zeros((1, n_k, number_of_states, 3)) if spinor else None
    kept_vectors = [[None] * n_k for _ in channels] if keep_vectors else None

    hubbard_matrices = potentials.get("hubbard")
    if hubbard_matrices is not None:
        if mode not in ("scalar", "spin_polarized"):
            raise NotImplementedError(
                "LDA+U bands are implemented for the scalar and collinear spin modes"
            )
        # Every backend applies the term: the device Bloch operators of the
        # k-point workers add P_k D P_k^H of a BlochHubbardHamiltonian
        # (acceleration/Hamiltonian/device_bloch.py).

    def with_hubbard(hamiltonians_k, k_point):
        """Add ``P_k D P_k^H`` of the SCF's ``D`` to every channel at ``k_point``."""
        if hubbard_matrices is None:
            return hamiltonians_k
        from ..Hamiltonian.hubbard import (
            HubbardOperator,
            build_hubbard_projectors,
            hubbard_replica_lattice,
        )
        from .kpoint_hubbard import BlochHubbardHamiltonian

        projectors = build_hubbard_projectors(
            system.grid,
            system.atoms,
            system.pseudopotentials,
            system.input.pseudopotentials,
            hubbard_replica_lattice(system),
            k_point=k_point,
        )
        return [
            BlochHubbardHamiltonian(
                hamiltonian, HubbardOperator(projectors, hubbard_matrices[channel])
            )
            for channel, hamiltonian in enumerate(hamiltonians_k)
        ]

    def hamiltonians(k_point):
        """One ``(operator, is_spinor)`` per channel at ``k_point``."""
        nonlocal_k, soc_k = _k_point_operators(system, k_point[None, :])[0]
        if mode == "scalar":
            return with_hubbard(
                [
                    KPointKohnShamHamiltonian(
                        system.negative_laplacian,
                        system.gradient,
                        potentials["effective_potential"],
                        nonlocal_k,
                        k_point,
                        # the kinetic term of the SCF (PARSEC_KPOINT_KINETIC)
                        bloch_laplacian=bloch_laplacian_or_none(system.grid, k_point),
                    )
                ],
                k_point,
            )
        if mode == "spin_polarized":
            return with_hubbard(
                [
                    KPointKohnShamHamiltonian(
                        system.negative_laplacian,
                        system.gradient,
                        potentials[name],
                        nonlocal_k,
                        k_point,
                        bloch_laplacian=bloch_laplacian_or_none(system.grid, k_point),
                    )
                    for name in ("potential_up", "potential_down")
                ],
                k_point,
            )
        extra = {}
        if mode == "soc_spin_polarized":
            extra["xc_delta"] = potentials["xc_delta"]
        elif mode == "noncollinear":
            extra["xc_field"] = potentials["xc_field"]
        return [
            _spinor_hamiltonian(
                system,
                k_point,
                nonlocal_k,
                soc_k,
                potentials["effective_potential"],
                **extra,
            )
        ]

    def solve(operator, attempt: int):
        """One diagonalization; later attempts use a larger CHEBDAV block and a new seed."""
        settings_now = eigval_settings
        if attempt and eigval_settings.initial_method == "chebdav":
            base = eigval_settings.chebdav
            settings_now = dataclasses.replace(
                eigval_settings,
                chebdav=dataclasses.replace(
                    base,
                    block_size=base.block_size + 2 * attempt,
                    random_seed=base.random_seed + attempt,
                ),
            )
        solution = solve_eigval(operator, number_of_states, settings=settings_now, state=None)
        values = np.asarray(solution.eigenvalues, dtype=float)
        vectors = np.asarray(solution.vectors)
        residual = np.asarray(operator @ vectors) - vectors * values[None, :]
        return values, vectors, float(np.max(np.linalg.norm(residual, axis=0)))

    can_retry = eigval_settings.initial_method == "chebdav"

    def solve_point(k_point, converter=None):
        """Bands of one k-point (all channels) with the CHEBDAV re-solve."""
        point_values = np.zeros((len(channels), number_of_states))
        point_residuals = np.zeros(len(channels))
        point_attempts = np.ones(len(channels), dtype=int)
        point_sigma = np.zeros((number_of_states, 3)) if spinor else None
        point_vectors = [] if keep_vectors else None
        for channel, hamiltonian in enumerate(hamiltonians(k_point)):
            # A device converter that can solve on the device (duck-typed, so that this module does
            # not import the acceleration package) keeps the whole first solve on the GPU; ARPACK
            # and DIAGLA stay host eigensolvers on the device operator.
            device_solve = (
                converter is not None
                and getattr(converter, "device_solver", False)
                and device_solver_supports(eigval_settings)
            )
            if device_solve:
                operator = None
            else:
                operator = (
                    converter.operator(hamiltonian).as_linear_operator()
                    if converter is not None
                    else hamiltonian.as_linear_operator()
                )
            best = None
            failure = None
            for attempt in range(max_attempts if can_retry else 1):
                try:
                    if device_solve:
                        trial = converter.eigensolve(hamiltonian, number_of_states, eigval_settings, attempt)
                    else:
                        trial = solve(operator, attempt)
                except FloatingPointError as error:
                    # A diverged Chebyshev recurrence (a bad spectral bound from an
                    # unlucky start vector): try again with another block size and seed.
                    failure = error
                    point_attempts[channel] = attempt + 1
                    continue
                point_attempts[channel] = attempt + 1
                if best is None or trial[2] < best[2]:
                    best = trial
                if trial[2] <= residual_tolerance:
                    break
            if best is None:
                raise FloatingPointError(
                    f"band k-point at {np.round(k_point, 6).tolist()}: every one of "
                    f"{point_attempts[channel]} diagonalization attempts diverged ({failure})"
                ) from failure
            values, vectors, point_residuals[channel] = best
            point_values[channel] = values
            if spinor:
                point_sigma = spin_expectation(vectors)
            if keep_vectors:
                host = vectors.get() if hasattr(vectors, "get") else vectors
                point_vectors.append(np.array(host, copy=True))
        return point_values, point_residuals, point_attempts, point_sigma, point_vectors

    indices = None
    residual_scope = None
    if symmetry is not None and mode == "scalar" and hubbard_matrices is None and not keep_vectors:
        # The k-points of a Gamma cell solved in sectors (acceleration/SCF/band_sectors.py): None where none splits.
        from ..acceleration.SCF.band_sectors import last_band_sector_record, solve_band_sectors

        found = solve_band_sectors(
            system,
            k_points,
            potential=potentials["effective_potential"],
            number_of_states=number_of_states,
            eigval_settings=eigval_settings,
            symmetry=symmetry,
            make_hamiltonian=lambda k_point, _potential: hamiltonians(np.asarray(k_point))[0],
            callback=callback,
            workers=workers,
            backend=backend,
        )
        if found is not None:
            eigenvalues[0], residuals[0] = found
            if last_band_sector_record().get("residuals_of_reported") and band_reported_residuals_setting() == "auto":
                residual_scope = "reported levels"
            # A CHEBDAV first solver re-solves, whole and with its attempts below, a k-point whose largest residual
            # in the sectors exceeds residual_tolerance, and keeps the better of the two (CHEBFF: never retried).
            again = [k for k in range(n_k) if can_retry and not residuals[0, k] <= residual_tolerance]
            if not again:
                return KPointBands(
                    channels=channels,
                    eigenvalues=eigenvalues,
                    residual_norms=residuals,
                    attempts=attempts,
                    residual_scope=residual_scope,
                )
            indices = again

    in_sectors = indices is not None
    if indices is None:
        indices = list(range(n_k))
    for task, (
        values,
        point_residuals,
        point_attempts,
        point_sigma,
        point_vectors,
    ) in map_tasks(
        lambda task, converter: solve_point(k_points[indices[task]], converter),
        len(indices),
        problem_size=system.grid.size,
        workers=workers,
        backend=backend,
    ):
        k_index = indices[task]
        if in_sectors:
            # after the solve in sectors: the better of the two, one attempt more
            point_attempts = point_attempts + 1
            if float(residuals[0, k_index]) <= float(point_residuals.max()):
                attempts[:, k_index] = point_attempts
                continue
        eigenvalues[:, k_index] = values
        if keep_vectors:
            for channel, vectors_k in enumerate(point_vectors):
                kept_vectors[channel][k_index] = vectors_k
        residuals[:, k_index] = point_residuals
        attempts[:, k_index] = point_attempts
        if spinor:
            sigma[0, k_index] = point_sigma
        if callback is not None:
            callback(
                k_index,
                n_k,
                values,
                float(point_residuals.max()),
                int(point_attempts.max()),
            )

    return KPointBands(
        channels=channels,
        eigenvalues=eigenvalues,
        residual_norms=residuals,
        attempts=attempts,
        spin_expectation=sigma,
        vectors=kept_vectors,
        residual_scope=residual_scope,
    )


__all__ = [
    "KPointBands",
    "MODES",
    "band_mode",
    "band_path_cartesian",
    "band_potentials",
    "load_scf_for_bands",
    "run_band_structure",
    "solve_kpoint_bands",
    "spin_expectation",
]
