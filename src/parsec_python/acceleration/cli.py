"""Command-line runner for the additive accelerated single-point package."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Output.parsec_output import domain_dry_run_lines
from parsec_python.models import SinglePointInput
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.MolecularDynamics.run import (
    restart_structure as dynamics_restart_structure,
    run_recorded_molecular_dynamics,
)
from parsec_python.Forces.position_symmetry import symmetrize_input_positions
from parsec_python.Relaxation.run import (
    move_flag_subgroup_setting,
    relaxation_header_lines,
    restart_structure,
    run_recorded_relaxation,
    run_single_point_with_forces,
    symmetry_decision,
    symmetry_fixed_atoms,
)
from parsec_python.cli import (
    load_restart_start,
    main as reference_main,
    output_file_conflict,
    output_files_for,
    refinement_progress,
    relaxation_exit_code,
    save_result_archive,
    save_self_consistent_soc_result_archive,
    start_output_files,
)
from parsec_python.Output.output_files import ReporterWithFiles, output_header_lines

from .Output import AcceleratedTextReporter
from .Output.post_scf import accelerated_band_structure, accelerated_dos, run_post_scf_outputs
from .backends.selection import resolve_backend
from .SCF.single_point import materialize_wavefunctions
from .driver import (
    _cupy_installed,
    hartree_boundary_check_seconds,
    ionic_backend,
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
)
from .models import BackendUnavailableError
from .SCF.self_consistent_soc import run_self_consistent_soc
from .resources import Resources, native_thread_information, resolve_resources


class _RunLog:
    def __init__(self, path: Path | None, quiet: bool = False) -> None:
        self.path = path
        self.quiet = quiet
        self._stream = None
        self.failure_reported = False

    def __enter__(self) -> "_RunLog":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, _type, value, _traceback) -> None:
        if value is not None:
            message = (
                "Calculation interrupted by user."
                if isinstance(value, KeyboardInterrupt)
                else f"Calculation failed: {type(value).__name__}: {value}"
            )
            try:
                self.write(message)
                self.failure_reported = True
            except OSError:
                pass
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def write(self, message: str = "") -> None:
        if not self.quiet:
            print(message, flush=True)
        if self._stream is not None:
            self._stream.write(message + "\n")
            self._stream.flush()


def _resolve_output(
    requested: Path | None,
    input_directory: Path,
    default_name: str,
) -> Path:
    if requested is None:
        return input_directory / default_name
    requested = requested.expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (input_directory / requested).resolve()


def _npz_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".npz" else path.with_suffix(path.suffix + ".npz")


def _symmetry_cache_directory(arguments: argparse.Namespace) -> Path | None:
    """Return the symmetry cache directory the command line named, if any.

    A first calculation of a structure is fastest without a cache, so none
    is read or written unless ``--symmetry-cache`` names its directory;
    ``--no-symmetry-cache`` spells that default out.  A relative directory
    is taken from the working directory, as it always was.
    """

    if arguments.symmetry_cache is None:
        return None
    return arguments.symmetry_cache.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the modular isolated real-space DFT solver. By default it runs "
            "on the GPU where one is usable and on the CPU otherwise (--device "
            "auto), on every GPU the process can see and with the CPUs it may "
            "use less four; --device, --gpus and --threads name them explicitly. "
            "On the GPU the Hamiltonian and the eigensolver run in CuPy and, "
            "with the native extension, finite-difference construction and "
            "Hartree in C++; results are in double precision either way."
        ),
    )
    parser.add_argument("input", type=Path, help="PARSEC input file, usually parsec.in")
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help=(
            "take the atom coordinates from this file instead of the input: a "
            "relax_restart.dat (its BFGS state continues a BFGS relaxation) or "
            "an .xyz file (last frame); Relax_Restart in the input reads "
            "relax_restart.dat beside it"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default=None,
        help=(
            "where the calculation runs: auto (default) takes the GPU if CuPy "
            "finds a usable one and the CPU otherwise; gpu requires a GPU and "
            "stops where there is none; cpu never touches a GPU. The variable "
            "PARSEC_DEVICE gives the same choice to a job script; parsec.out "
            "says what ran"
        ),
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpus",
        default=None,
        metavar="N|all",
        help=(
            "how many GPUs of this node to use: all (default) is every GPU the "
            "process can see, which in a scheduler job is what the job was "
            "given; N takes the first N of them. Naming a number asks for a "
            "GPU run"
        ),
    )
    gpu_group.add_argument(
        "--gpu-ids",
        default=None,
        metavar="LIST",
        help=(
            "the GPUs to use by index, such as 0,2: indices count the GPUs the "
            "process can see (inside CUDA_VISIBLE_DEVICES, which is left alone)"
        ),
    )
    parser.add_argument(
        "--threads",
        default=None,
        metavar="N|all|auto",
        help=(
            "CPU threads of the C++/OpenMP kernels: auto (default) follows "
            "OMP_NUM_THREADS and otherwise takes the CPUs the process may use "
            "less four, all takes every one of them, N names the number. The "
            "CPUs are those of the allocation in a scheduler job, not those of "
            "the machine"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "scipy", "native", "cupy"),
        default="auto",
        help=(
            "which implementation executes, for controlled comparisons; "
            "--device is the usual choice. auto (default) follows --device: on "
            "the GPU it combines native setup/Hartree with the CuPy "
            "Hamiltonian/eigensolver, on the CPU it takes native C++ or, "
            "without the extension, SciPy. Explicit scipy and native (CPU) and "
            "cupy (GPU) never fall back; every selected component and fallback "
            "is recorded"
        ),
    )
    parser.add_argument(
        "--symmetry",
        choices=("auto", "on", "off"),
        default=None,
        help=(
            "symmetry policy: auto (default) detects exact supported "
            "operations and falls back safely, on requires a nontrivial "
            "usable symmetry, and off forces full-grid calculations; "
            "Ignore_Symmetry=true selects off unless this option is given"
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--symmetry-cache",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help=(
            "read and write exact-key symmetry maps, representation "
            "operators and Hartree boundary geometry in DIRECTORY; it pays "
            "only for repeated calculations of the same structure and grid, "
            "and no directory is read or written unless it is named here"
        ),
    )
    cache_group.add_argument(
        "--no-symmetry-cache",
        action="store_true",
        help="use no persistent symmetry cache (the default)",
    )
    parser.add_argument(
        "--pp-dir",
        type=Path,
        default=None,
        help="additional pseudopotential directory searched before the input directory",
    )
    parser.add_argument(
        "--esdf-lenient",
        action="store_true",
        help=(
            "read the input as PARSEC's ESDF does: ignore labels this reader does not know and take the "
            "first value of a label given twice, with a warning (default: refuse both)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result .npz path; defaults beside parsec.in",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="text log path; defaults to parsec.out beside parsec.in",
    )
    parser.add_argument(
        "--kpoint-workers",
        default=None,
        metavar="N|auto",
        help=(
            "worker threads for the k-points of a periodic SCF and for the band path "
            "(reference solver; default PARSEC_KPOINT_WORKERS, else auto)"
        ),
    )
    parser.add_argument(
        "--kpoint-pools",
        default=None,
        metavar="P|auto",
        help=(
            "with --mpi: pools of ranks the k-points and spin channels are dealt to, each basis shared "
            "by the ranks of its pool (default PARSEC_KPOINT_POOLS, else auto: the best balanced count "
            "whose memory fits the GPUs)"
        ),
    )
    parser.add_argument(
        "--bands-from",
        type=Path,
        default=None,
        metavar="ARCHIVE.npz",
        help=(
            "skip the SCF and compute the bandstruc bands from the converged potential in a "
            "previous run's result archive (reference solver; log: parsec_bands.out)"
        ),
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "run as one of several MPI ranks (start them with mpirun or srun, one per GPU): "
            "the k-points and spin channels are dealt to pools of ranks, the basis of each shared "
            "by the GPUs of its pool (--kpoint-pools); spin-orbit and non-collinear bases are split "
            "by blocks of grid rows over all the GPUs. Every rank runs the same calculation; only "
            "rank 0 writes the log, the archive and the data files. Same as PARSEC_MPI=1"
        ),
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="do not write parsec_python_results.npz",
    )
    parser.add_argument(
        "--save-wavefunctions",
        action="store_true",
        help="include wavefunctions in the result archive",
    )
    parser.add_argument(
        "--mlff",
        nargs="?",
        const="",
        default=None,
        metavar="MODEL",
        help=(
            "relax the input structure with a machine-learned force field first "
            "(Initial_Structure: mlff); MODEL as MLFF_Model, for example uma-s-1p2, "
            "mace-mp-medium-mpa-0 or a file.model; without MODEL the input's or uma-s-1p2"
        ),
    )
    parser.add_argument(
        "--energy-refinement",
        choices=("b3lyp", "none"),
        default=None,
        help=(
            "after the SCF, evaluate the exchange-correlation energy of this "
            "hybrid functional once on the converged orbitals and report the "
            "refined total energy (PP-PBE(B3LYP) with a PBE input; ARES "
            "Lrefine); overrules Energy_Refinement of the input, none "
            "switches it off"
        ),
    )
    parser.add_argument(
        "--profile-operator",
        action="store_true",
        help=(
            "run an opt-in synchronized microprofile of finite-difference, "
            "local ionic, and nonlocal ionic Hamiltonian actions"
        ),
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=3,
        help="component-profile repeats averaged when --profile-operator is used",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse/validate input and resolve backend without constructing the grid",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress console progress while retaining the text log",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help=(
            "submit to the local warmed worker, auto-starting it if needed; "
            "handled by the folder-local launcher before ordinary parsing"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise runtime exceptions with a traceback",
    )
    parser.epilog = (
        "Resident controls: main.py --resident-start, --resident-status, "
        "or --resident-stop. Each request creates fresh DFT/SCF state while "
        "reusing only process and accelerator runtime state."
    )
    return parser


def _visible_gpus(resources: Resources) -> int:
    """Count the GPUs of this process and hold a named number or list against them."""

    from .backends.cupy import CuPyUnavailableError, cupy_device_count

    try:
        visible = cupy_device_count()
    except CuPyUnavailableError as error:
        raise BackendUnavailableError(
            f"a GPU run was requested ({resources.device_source}) but no GPU is usable: {error}"
        ) from error
    resources.check_gpus(visible)
    return visible


def _resource_report(resources: Resources, system) -> list[str]:
    """The lines of ``parsec.out`` that say what the calculation runs on."""

    info = system.backend_info
    gpus = visible = None
    if info.selected == "cupy":
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import cupy_device_count, require_cupy

        cp, _ = require_cupy()
        visible = cupy_device_count()
        gpus = selected_device_ids(int(cp.cuda.Device().id))
    return resources.report_lines(
        info.selected,
        fallback_reasons=tuple(info.fallback_reasons),
        gpus=gpus,
        visible_gpus=visible,
        native=native_thread_information(),
    )



def _accelerated_self_consistent_soc_eligible(translation) -> bool:
    """Whether this input matches the one self-consistent-SOC shape the
    accelerated path implements: Gamma-point only, not combined with
    collinear spin polarization, and (on the default routines) with forces, relaxation or molecular dynamics only
    under ``PARSEC_SOC_DEVICE_FORCES=auto`` (the default: the device forces of
    :mod:`parsec_python.acceleration.Forces.spinor_device`; ``off`` hands them to the reference solver, as before;
    the Fortran-derived routines are eligible either way and :func:`_reference_only_features` names their moving
    atoms under ``off``).

    See :mod:`parsec_python.acceleration.SCF.self_consistent_soc`'s module
    docstring.  Combined with spin polarization or a periodic
    ``Boundary_Conditions`` always delegates to the reference CPU solver
    (:mod:`parsec_python.SCF.self_consistent_soc_spin_polarized`/
    ``kpoints_soc`` have no CuPy counterpart yet).
    """

    problem = translation.problem
    # (the Fortran-derived routines, as before the merge: moving atoms are named by _reference_only_features instead)
    fortran = getattr(getattr(translation, "routines", None), "fortran", False)
    return (
        problem.scf.self_consistent_spin_orbit
        and not problem.scf.spin_polarized
        and not problem.scf.noncollinear
        and problem.periodic_cell is None
        and (fortran or not _asks_for_ionic_motion(translation) or _soc_device_forces(translation))
    )


def _run_soc_ionic(
    arguments,
    translation,
    system,
    *,
    reporter,
    files,
    log,
    soc_symmetry: bool,
    force_symmetry: bool,
    relax_restart,
    dynamics_restart,
    input_directory,
    archive_path,
    log_path,
    started: float,
    scf_started: float,
) -> int:
    """The forces, relaxation or molecular dynamics of a self-consistent spin-orbit cluster on the device
    (``PARSEC_SOC_DEVICE_FORCES=auto``): the ionic loops of the reference solver's spinor path
    (``Relaxation.spinor_backend``) with the device spinor loop and forces of
    :func:`.SCF.self_consistent_soc.soc_ionic_backend`, each SCF iteration and structure reported with the
    spin-orbit tables, the archive of a spin-orbit run with its forces and structures; returns the exit status."""

    from parsec_python.cli import _SpinorReporter

    from .SCF.self_consistent_soc import soc_ionic_backend

    # the Fortran-derived routines: the forces of Forces.spin_orbit_fortran, their ionic loop and their archive
    fortran = bool(getattr(getattr(translation, "routines", None), "fortran", False))
    # the spin-orbit tables of the reference reporter, and the files of every iteration and structure
    soc_reporter = _SpinorReporter(ReporterWithFiles(reporter.reference, files))

    def backend_of(extrapolation: str = "none"):
        return soc_ionic_backend(
            # --symmetry off, and Ignore_Symmetry without a mode on the command line, keep the whole grid
            symmetry=soc_symmetry,
            # --symmetry on: PARSEC_SOC_SECTORS=auto takes the sector at every size
            forced=arguments.symmetry == "on",
            note=log.write,
            # (the ionic loop of the Fortran-derived routines takes no extrapolation, Relaxation.fortran_spinor_backend)
            potential_extrapolation="none" if fortran else extrapolation,
            fortran=fortran,
        )

    outcome = None
    dynamics = None
    if translation.relaxation is not None:
        outcome = run_recorded_relaxation(
            translation,
            system,
            backend_of(translation.potential_extrapolation),
            soc_reporter,
            log.write,
            directory=input_directory,
            restart=relax_restart,
            symmetrize=force_symmetry,
        )
        result = outcome.result
        forces = outcome.records[-1].forces
    elif translation.molecular_dynamics is not None:
        dynamics = run_recorded_molecular_dynamics(
            translation,
            system,
            backend_of(translation.potential_extrapolation),
            soc_reporter,
            log.write,
            directory=input_directory,
            restart=dynamics_restart,
        )
        result, forces = dynamics.result, dynamics.forces
    else:
        result, forces = run_single_point_with_forces(
            system, backend_of(), soc_reporter, symmetrize=force_symmetry
        )
    from .Forces.spinor_device import release_device_spinors

    # (the last structure's device spinors and subspace are not needed again)
    release_device_spinors(result)
    result.eigensolver_state = None
    log.write(f" Pre-SCF setup/reporting wall time [sec] : {scf_started - started:11.6f}")
    log.write(f" Total accelerated Python wall time [sec] : {time.perf_counter() - started:11.2f}")
    if not arguments.no_archive:
        # the archive of the reference solver's spinor loops: that of a spin-orbit run on the default routines, the
        # general one on the Fortran-derived routines
        save = save_result_archive if fortran else save_self_consistent_soc_result_archive
        saved = save(
            archive_path,
            result,
            include_wavefunctions=(arguments.save_wavefunctions or translation.output_all_states),
            **{
                name: value
                for name, value in (("forces", forces), ("relaxation", outcome), ("dynamics", dynamics))
                if value is not None
            },
        )
        log.write(f"Result archive: {saved}")
    # (the moved structures keep the grid)
    files.finish(result, system.grid, translation.problem)
    log.write(f"Text log: {log_path}")
    if dynamics is not None:
        return 3 if dynamics.scf_failed else 0
    return relaxation_exit_code(
        result,
        outcome,
        manual=(translation.relaxation is not None and translation.relaxation.method == "manual"),
    )


def _soc_device_forces(translation) -> bool:
    """Whether the forces, relaxation or molecular dynamics of a self-consistent spin-orbit cluster (neither
    spin-polarized nor non-collinear) run on the accelerated path (``PARSEC_SOC_DEVICE_FORCES=auto``, either routine
    set, one process, without ``Create_DOS``; an unknown value of the switch is refused before the SCF by
    :func:`main`)."""

    from .Forces.spinor_device import soc_device_forces_setting
    from .Parallel import mpi

    problem = translation.problem
    if not (
        problem.scf.self_consistent_spin_orbit
        and not problem.scf.spin_polarized
        and not problem.scf.noncollinear
        and problem.periodic_cell is None
    ):
        return False
    if getattr(translation, "dos", None) is not None or mpi.requested():
        # Create_DOS of the last structure, and --mpi: the reference solver's spinor loops (the input reader refuses
        # Write_Potential_Field with spin-orbit coupling)
        return False
    try:
        return soc_device_forces_setting() == "auto"
    except ValueError:
        return False


def _asks_for_ionic_motion(translation) -> bool:
    """Whether the input asks for forces, a relaxation or molecular dynamics.

    The CuPy spinor driver has them for an unpolarized cluster under ``PARSEC_SOC_DEVICE_FORCES=auto``
    (:func:`_soc_device_forces`); the reference solver has them for every spin-orbit and non-collinear input.
    """

    return bool(
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or translation.compute_forces
    )


def _resolve_symmetry_mode(
    requested_symmetry: str | None,
    ignore_symmetry: bool,
    *,
    accelerated_self_consistent_soc: bool,
    soc_sectors: str = "off",
) -> str:
    """Resolve ``--symmetry``/``Ignore_Symmetry`` to an effective mode.

    Self-consistent SOC on the accelerated path always forces "off" for the
    scalar machinery the solver is prepared with: the orbital-sector symmetry
    decomposition (``CuPySymmetrySCFEigensolver``/``SymmetrySCFReducer``)
    assumes real-valued, spin-diagonal 1-D irreps, but the spin-orbit L.S
    cross term (``lzsz``/``lsxy``) mixes spin channels and breaks that
    assumption, and the spinor loop keeps its Hartree/XC/mixing on the whole
    grid.  Its own reduction is the double-group sectors of the spinor loop
    (``PARSEC_SOC_SECTORS=kramers`` or ``on``, and ``auto``, the default, from
    the floor of :func:`.SCF.soc_sectors.resolve_soc_sectors` on and at every
    size under ``--symmetry on``; ``soc_sectors``,
    :mod:`.SCF.soc_sectors`), which the caller enables unless the run uses no
    symmetry (``--symmetry off``, or ``Ignore_Symmetry`` without a mode on the
    command line, :func:`_soc_symmetry_choice`).  "auto"/unset keeps whatever the
    switch gives; "on" explicitly promises a nontrivial reduction, so with
    ``PARSEC_SOC_SECTORS=off`` it is rejected with a ``ValueError`` that names
    the switch (the caller refuses it too for a cluster without a two-fold
    rotation or mirror, where the sectors cannot be used).
    """

    mode = (
        requested_symmetry
        if requested_symmetry is not None
        else ("off" if ignore_symmetry else "auto")
    )
    if accelerated_self_consistent_soc:
        if requested_symmetry == "on" and soc_sectors == "off":
            raise ValueError(
                "self-consistent spin-orbit coupling on the accelerated path "
                "reduces by symmetry only in the double-group sectors of "
                "PARSEC_SOC_SECTORS=kramers or on, so --symmetry on cannot be "
                "satisfied with PARSEC_SOC_SECTORS=off; set PARSEC_SOC_SECTORS "
                "or omit --symmetry"
            )
        return "off"
    return mode


def _soc_symmetry_choice(
    symmetry_mode: str,
    requested_symmetry: str | None,
    ignore_symmetry: bool,
    soc_sectors: str,
) -> tuple[str, bool, str]:
    """The symmetry of a self-consistent SOC run on the accelerated path: the mode the solver is prepared with
    (always ``"off"``, :func:`_resolve_symmetry_mode`, which refuses ``--symmetry on`` with
    ``PARSEC_SOC_SECTORS=off``), whether the spinor loop may use the double-group sectors of the switch, and the
    mode the report is written with.

    ``symmetry_mode`` is the mode of the run before this choice (the command line where it overrides
    ``Ignore_Symmetry``, as for every other run): the sectors are allowed unless it is ``"off"``, and the report
    then prints PARSEC 1.5's block of the point group rather than the lines of ``Ignore_Symmetry``.
    """

    resolved = _resolve_symmetry_mode(
        requested_symmetry,
        ignore_symmetry,
        accelerated_self_consistent_soc=True,
        soc_sectors=soc_sectors,
    )
    allowed = symmetry_mode != "off"
    return resolved, allowed, (symmetry_mode if allowed else resolved)


def _soc_orbital_symmetry(system, soc_sectors: str, symmetry_on: bool):
    """The backend record of a self-consistent SOC run whose input allows symmetry, with the
    ``orbital_symmetry`` of its spinor loop: the double-group sectors of ``PARSEC_SOC_SECTORS`` (or why the
    whole grid stays), or the whole grid under ``off``.  ``--symmetry on`` (``symmetry_on``) with a cluster
    whose sectors cannot be used is refused (``ValueError``) before the SCF, as the scalar route refuses it.
    """

    info = system.backend_info
    route, floor = soc_sectors, ""
    if soc_sectors == "auto":
        # PARSEC_SOC_SECTORS=auto (the default): the Kramers sector from the floor of the whole-grid spinor basis on,
        # at every size under --symmetry on
        from .SCF.soc_sectors import resolve_soc_sectors

        route, floor = resolve_soc_sectors(soc_sectors, system, forced=symmetry_on)
    if route == "off":
        text = (
            "full grid (self-consistent spin-orbit coupling: PARSEC_SOC_SECTORS=off)"
            if not floor
            else "full grid (self-consistent spin-orbit coupling: PARSEC_SOC_SECTORS=auto, the whole-grid spinor "
            "basis below the floor of the sectors)"
        )
    else:
        from .SCF.soc_sectors import soc_orbital_symmetry

        text, reason = soc_orbital_symmetry(system.grid, system.atoms, route)
        if soc_sectors == "auto":
            text = text.replace(f"PARSEC_SOC_SECTORS={route}", "PARSEC_SOC_SECTORS=auto")
        if reason and symmetry_on:
            raise ValueError(
                f"symmetry='on' requested, but the spin-orbit sectors of PARSEC_SOC_SECTORS={soc_sectors} "
                f"cannot be used: {reason}"
            )
    details = tuple(
        (key, text if key == "orbital_symmetry" else value) for key, value in info.details
    )
    if not any(key == "orbital_symmetry" for key, _ in details):
        details = details + (("orbital_symmetry", text),)
    return replace(info, details=details)


def _reference_only_solver_options(problem) -> list[str]:
    """Eigensolver options of the PARSEC 1.5 round that the device eigensolvers do not carry."""

    eigensolver = problem.eigensolver
    features = []
    if getattr(eigensolver, "first_filter_residual_test", "off") != "off":
        features.append("PARSEC 1.5's residual-driven CHEBFF (FF_Residual_Test)")
    # Smart_Subspace (the lower-bound rule of SUBSPACE) and Bounds_Method: lanczos
    # with Bounds_Iter (PARSEC 1.5's upper bound) run on the device filters too
    # (Eigensolvers.subspace.subspace_filter_bounds and
    # Eigensolvers.spectral_bounds.lanczos_bound_from_ritz, since 2026-11-08).
    if (
        getattr(eigensolver, "scaled_subspace_filter", False)
        or getattr(eigensolver, "lanczos_bounds", "parsec") == "lanczos_full"
    ):
        features.append(
            "PARSEC 1.5's SUBSPACE rules (Chebyshev_Scaled, Bounds_Method: lanczos_full)"
        )
    if getattr(eigensolver, "reorthogonalization_threshold", 0.1) != 0.1:
        features.append(
            "another Gram-Schmidt reorthogonalization threshold "
            "(Reorthogonalization_Threshold)"
        )
    return features


def _symmetrized_cell_in_sectors(problem) -> bool:
    """Whether the accelerated driver runs ``Symmetrize_Charge_Density`` of this periodic cell
    (``PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto``): an unpolarized bulk cell at the Gamma point that the route of the
    sectors takes (:func:`.driver.periodic_sector_refusal`); the rest keeps the reference solver."""

    from .driver import periodic_sector_refusal

    if bool(getattr(problem.scf, "spin_polarized", False)):
        # (a spin cell with the option keeps the reference 1x1x1 k-point driver: SCF/gamma_spin.py)
        return False
    try:
        return periodic_sector_refusal(problem) is None
    except (AttributeError, TypeError, ValueError):
        return False


def _reference_only_features(
    translation,
    *,
    accelerated_backend_selected: bool,
    device_eigensolver_selected: bool = False,
    hybrid_on_device: bool = False,
    gamma_spin_on_device: bool = False,
    general_grid_on_device: bool = False,
) -> list[str]:
    """Input features only the reference :mod:`parsec_python.cli` dispatches.

    The accelerated driver runs a scalar Gamma-point SCF (forces, relaxation and molecular dynamics of isolated
    systems only), spin-polarized
    where the system is isolated and carries no spin-orbit species, plus (when ``accelerated_backend_selected`` and the input is
    Gamma-point-only self-consistent SOC without spin polarization --
    :func:`_accelerated_self_consistent_soc_eligible`) the CuPy spinor
    Hamiltonian/eigensolver in
    :mod:`parsec_python.acceleration.SCF.self_consistent_soc`.  Without this
    check every other SOC/spin/k-point flag would be silently ignored.
    ``device_eigensolver_selected``: the backend resolves to the CuPy
    eigensolvers, which run ``Eigensolver: arpack`` on the device
    (:mod:`parsec_python.acceleration.Eigensolvers.arpack`).
    ``hybrid_on_device``: the input has a self-consistent hybrid functional
    and the backend resolves to CuPy in one process, where the device Fock
    loop runs it (:mod:`parsec_python.acceleration.SCF.hybrid`); otherwise
    the reference solver runs it.
    ``gamma_spin_on_device``: a collinear spin-polarized bulk cell at the
    Gamma point that the accelerated Gamma sectors run
    (:func:`_gamma_spin_on_device`, ``PARSEC_GAMMA_SPIN_SECTORS``); otherwise
    the reference solver's 1x1x1 k-point driver runs it.
    ``general_grid_on_device``: the backend resolves to CuPy (:func:`_cupy_backend_selected`), where a
    Gamma-point cell on Fortran's general grid takes the accelerated driver
    (``PARSEC_GENERAL_GRID_ACCELERATED``); otherwise the reference solver runs it.
    """

    problem = translation.problem
    # The routines of the run (parsec_python.Input.routines): the routes of
    # the Fortran parity round for the Fortran-derived ones; Andrew Grieder's
    # accelerated path also takes the Gamma point named as a one-point grid
    # and his grids of every cell shape.
    fortran = getattr(getattr(translation, "routines", None), "fortran", False)
    features = []
    if getattr(problem.scf, "is_hybrid", False) and not hybrid_on_device:
        # The device Fock loop (acceleration.SCF.hybrid) needs the CuPy
        # backend in one process; on the CPU backends and under --mpi the
        # reference solver runs the Fock loop of parsec_python.SCF.hybrid,
        # with its single point, forces and relaxation.
        features.append(
            f"the self-consistent hybrid functional (Correlation_Type: {problem.scf.xc_functional})"
        )
    if problem.scf.noncollinear:
        features.append("non-collinear magnetism (Non_Collinear_magnetism)")
    elif problem.scf.self_consistent_spin_orbit:
        if problem.scf.spin_polarized:
            features.append(
                "self-consistent spin-orbit coupling combined with spin "
                "polarization (SO_from_scratch + Spin_Polarization)"
            )
        elif problem.periodic_cell is not None:
            features.append(
                "periodic self-consistent spin-orbit coupling "
                "(SO_from_scratch with a periodic Boundary_Conditions)"
            )
        elif not accelerated_backend_selected:
            # (the Fortran-derived routines name the moving atoms below, as before the merge)
            features.append(
                "forces, relaxation or molecular dynamics with self-consistent spin-orbit coupling (SO_from_scratch)"
                if _asks_for_ionic_motion(translation) and not fortran
                else "self-consistent spin-orbit coupling (SO_from_scratch)"
            )
        # else: Gamma-point-only self-consistent SOC on a resolved CuPy
        # backend -- accelerated, not reference-only.
    elif problem.scf.spin_polarized and not gamma_spin_on_device and (
        problem.periodic_cell is not None
        or any(
            specification.spin_orbit
            for specification in problem.pseudopotentials.values()
        )
    ):
        features.append(
            "spin polarization of a periodic system or of spin-orbit species"
        )
    if (
        problem.has_kpoint_sampling
        and (fortran or not _gamma_point_only(problem))
        and not (
            (problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear)
            and problem.periodic_cell is not None
        )
    ):
        features.append("k-point sampling (Monkhorst-Pack or explicit list)")
    if problem.bands is not None and not accelerated_band_structure(translation):
        # After a Gamma-point SCF of a scalar cell the bands run after the
        # accelerated SCF (acceleration/Output/post_scf.py).
        features.append("band structure along a k-path (bandstruc)")
    if fortran and (problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear) and (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    ) and not (accelerated_backend_selected and _soc_device_forces(translation)):
        # The spin-orbit forces (Forces.spin_orbit_fortran) and the spinor ionic
        # loop exist on the reference path only, but for an unpolarized cluster on
        # CuPy under PARSEC_SOC_DEVICE_FORCES=auto (their device form,
        # Forces/spinor_device.py).  (Andrew Grieder's routines name
        # them above and route them with _accelerated_self_consistent_soc_eligible.)
        features.append(
            "forces, relaxation or molecular dynamics with spin-orbit coupling "
            "or non-collinear magnetism"
        )
    if (
        problem.periodic_cell is not None
        and (
            getattr(translation, "compute_forces", False)
            or getattr(translation, "relaxation", None) is not None
            or getattr(translation, "molecular_dynamics", None) is not None
        )
        and getattr(translation, "energy_refinement", None) is not None
    ):
        # The accelerated Gamma-point path of a periodic cell has its forces
        # and moved atoms (Forces/periodic_device.py); the energy refinement
        # of the reference path follows the forces there (k-points, spin,
        # wires and slabs, the general grid are listed above and below).
        features.append(
            "forces, relaxation or molecular dynamics of a periodic cell "
            "together with Energy_Refinement"
        )
    from .SCF.periodic_routes import moving_cell_options_reference_only

    features.extend(moving_cell_options_reference_only(translation))
    # The seven field loops (acceleration.SCF.polarizability), the potential
    # of point charges and the symmetrized density (acceleration.SCF.
    # density_symmetry) run on the accelerated SCF of a cluster; the
    # reference solver keeps those of a periodic cell (its space group),
    # except a bulk cell at the Gamma point that the route of the sectors
    # takes, whose density the space group averages on the orbits of its
    # sectors (PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto, since 2026-11-19).
    # A periodic spinor run (self-consistent spin-orbit coupling or
    # non-collinear magnetism) is named above already, and the reference
    # solver averages its density too (the wedge of PARSEC_SOC_KPOINT_WEDGE,
    # the Fortran-derived routines): no second or third reason for it.
    periodic_spinor = problem.periodic_cell is not None and bool(
        problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear
    )
    if (
        getattr(problem.scf, "symmetrize_density", False)
        and problem.periodic_cell is not None
        and not periodic_spinor
    ):
        from .SCF.density_symmetry import symmetrized_density_sectors_setting

        if symmetrized_density_sectors_setting() == "off" or not _symmetrized_cell_in_sectors(problem):
            features.append(
                "charge-density symmetrization of a periodic cell (Symmetrize_Charge_Density)"
            )
    if problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear:
        # The accelerated spinor loop (SCF.self_consistent_soc) has none of them.
        output_files = getattr(translation, "output_files", None)
        for present, name in (
            (getattr(translation, "polarizability_field", None) is not None, "Polarizability"),
            (bool(getattr(problem, "point_charges", ())), "Add_Point_Charges"),
            (
                getattr(problem.scf, "symmetrize_density", False) and not periodic_spinor,
                "Symmetrize_Charge_Density",
            ),
            (
                output_files is not None
                and output_files.output_data
                and output_files.intermediate_density,
                "Save_Intermediate_Charge_Density",
            ),
            (
                getattr(problem.grid, "domain_shape", "sphere") in ("ellipsoid", "cylinder"),
                "Cluster_Domain_Shape ellipsoid or cylinder",
            ),
        ):
            if present:
                features.append(f"{name} with spin-orbit coupling or non-collinear magnetism")
    hubbard_species = [
        specification
        for specification in problem.pseudopotentials.values()
        if getattr(specification, "has_hubbard", False)
    ]
    # The accelerated driver holds LDA+U (J = 0) in its device Hamiltonians
    # and forms the TS sums on a device, for clusters, unpolarized and
    # spin-polarized (acceleration.Hamiltonian.hubbard_device,
    # acceleration.Energy.dispersion_device).  The rest stays here by name.
    # A bulk cell at the Gamma point takes the accelerated driver too; its
    # k-point, spin, wire/slab and force runs go to the reference solver by
    # their own entries, whose k-point drivers apply the LDA+U term and form
    # the TS sums on their device workers.
    if any(float(item.hubbard_j) != 0.0 for item in hubbard_species):
        # PARSEC's J term is not symmetric; the device eigensolvers apply
        # symmetric separable terms only.
        features.append("LDA+U with ldaplusu_j != 0")
    if (
        hubbard_species
        and problem.periodic_cell is not None
        and getattr(problem.grid, "periodicity", 3) in (1, 2)
    ):
        # Validated on the reference wire/slab loop only (the replicas along
        # the periodic vectors of upot.F90), whatever route the wire or slab
        # itself takes.
        features.append("LDA+U of a wire or slab (ldaplusu_u)")
    if getattr(problem.scf, "van_der_waals", False):
        if problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear:
            # The accelerated spinor loop has no TS correction.
            features.append(
                "the Tkatchenko-Scheffler correction with spin-orbit coupling or "
                "non-collinear magnetism (Van_Der_Waals)"
            )
    # Eigensolvers of the reference solver only.  The accelerated SCF loop
    # is the shared one, so Convergence_Criterion_Approach, Dynamic_Diag_Tol
    # (the tolerance of each iteration reaches the device CHEBDAV first
    # solves and ARPACK) and Fix_Number_Eigenvalues (no buffer; symmetry
    # sectors keep their counts) run there; Eigensolver: none is the setup
    # of the accelerated driver.  ARPACK runs on the device where the CuPy
    # eigensolvers are selected, but not with spin-orbit coupling (the
    # spinor eigensolver is CHEBFF).  DIAGLA is the reference solver's.
    eigensolver = problem.eigensolver
    if eigensolver.method == "arpack" and (
        not device_eigensolver_selected
        or problem.scf.self_consistent_spin_orbit
        or problem.scf.noncollinear
        # The device ARPACK is validated on clusters; a periodic cell keeps
        # the reference ARPACK (on the device operator of the k-point workers).
        or problem.periodic_cell is not None
    ):
        features.append("the ARPACK eigensolver (Eigensolver: arpack)")
    elif eigensolver.method == "diagla":
        features.append("the DIAGLA eigensolver (Eigensolver: diagla)")
    # Eigensolver options of the PARSEC 1.5 round that only the readable
    # eigensolvers carry (a Vector_Rng_Seed start basis, Subspace_Buffer_Size
    # 0 and the degree decrease of Chebyshev_Degree_Decrease_After run on the
    # device too).
    features.extend(_reference_only_solver_options(problem))
    # Pseudopotential set-up: the double grid (acceleration/V_ion/double_grid.py),
    # Old_Interpolation_Format (the native and CuPy local builders and the
    # device local forces) and, for a periodic cell, the Fourier filters and
    # the double grid of the readable periodic preparation run accelerated.
    from .V_ion.setup_options import reference_only_setup_features

    features.extend(reference_only_setup_features(problem))
    output_files = getattr(translation, "output_files", None)
    if (
        output_files is not None
        and output_files.output_data
        and output_files.intermediate_density
        and problem.periodic_cell is not None
    ):
        # The accelerated SCF of a cluster hands its fields to parsec.dat
        # (acceleration.Output.iteration_fields).
        features.append(
            "parsec.dat after every SCF iteration of a periodic cell "
            "(Save_Intermediate_Charge_Density)"
        )
    if (
        output_files is not None
        and output_files.output_data
        and output_files.intermediate_eigenvalues
        and problem.has_kpoint_sampling
    ):
        # The k-point levels of every iteration come from the reference
        # drivers' iteration fields.
        features.append(
            "the k-point levels of every SCF iteration in eigen.dat "
            "(Save_Intermediate_Eigenvalues)"
        )
    if getattr(translation, "restart_run", False) and (
        getattr(problem.scf, "self_consistent_spin_orbit", False)
        or getattr(problem.scf, "noncollinear", False)
    ):
        # The spin-orbit loops take the stored potentials on the reference path.
        features.append("Restart_Run of a spin-orbit calculation")
    if getattr(translation, "restart_from_wfndat", False) and getattr(
        translation, "restart_run", False
    ):
        # A cluster seeds the full-grid eigensolver with the stored states
        # (acceleration.SCF.restart); sectors take no stored subspace.
        from .driver import single_sector_setting

        if problem.periodic_cell is not None:
            features.append(
                "a restart of a periodic cell from the legacy wfn.dat (Restart_From_Wfndat)"
            )
        elif single_sector_setting() == "on":
            features.append(
                "a restart from the legacy wfn.dat (Restart_From_Wfndat) with "
                "PARSEC_CUPY_SINGLE_SECTOR=on, whose one sector takes no stored subspace"
            )
    # Post-processing of the reference CLI (parsec_python.SCF.nscf,
    # parsec_python.Output.dos) and its reciprocal-space Hartree solve.
    if getattr(translation, "nscf", None) is not None:
        features.append("a non-self-consistent run (nscf_kpoints)")
    if getattr(translation, "dos", None) is not None and not accelerated_dos(translation):
        # The DOS of a cluster follows the accelerated SCF (Output/post_scf.py).
        features.append("the density of states (Create_DOS)")
    # Calc_Hartree_G, Periodic_Local_Potential: g_space and Zero_Average_Vxc of a
    # Gamma-point 3-D cell run on the accelerated driver: the reference periodic
    # preparation (G-space potential, alpha term), the Hartree solve of
    # Hartree.pbc (on the device: acceleration/Hartree/periodic_device.py) and the
    # shifted XC potential (V_xc/device_xc.py::ZeroAverageXC); with k-points,
    # spin or forces the entries above route them, and with --mpi this:
    from .SCF.periodic_routes import cell_options_reference_only

    features.extend(cell_options_reference_only(translation))
    if getattr(translation, "gw", None) is not None:
        # parsec_python.Output.berkeleygw after the reference SCF or NSCF run.
        features.append("the BerkeleyGW files WFN, RHO and VXC (Output_GW)")
    if problem.periodic_cell is not None and getattr(problem.grid, "periodicity", 3) in (1, 2):
        # Boundary_Conditions wire or slab (parsec_python.SCF.wire_slab): a
        # Gamma-point single point runs on the accelerated driver (the reference
        # preparation and Hartree plan, the device Hartree solve); k-points,
        # spin and forces are routed above.
        from .SCF.periodic_routes import wire_slab_reference_only

        features.extend(wire_slab_reference_only(translation))
    elif problem.periodic_cell is not None and fortran:
        from parsec_python.Grid.pbc import uses_general_periodic_grid

        if uses_general_periodic_grid(problem.periodic_cell, problem.grid):
            # A Gamma-point cell on that grid runs on the accelerated driver (PARSEC_GENERAL_GRID_ACCELERATED=auto,
            # since 2026-11-23) unless one of these keeps it with the reference solver; under off every such cell.
            from .SCF.periodic_routes import general_grid_reference_only

            features.extend(general_grid_reference_only(translation, on_device=general_grid_on_device))
    if problem.periodic_cell is None and getattr(problem.hartree, "cg_criterion", 1) == 2:
        # PARSEC 1.5's Hartree_CG_Criterion 2 (the CG stops against ||b||): the readable CG
        # (Hartree.poisson.cg_stopping_threshold) and the device CG of wires, slabs and cells
        # (Hartree/periodic_device.py) have it; the CG backends of the accelerated cluster
        # driver (CuPy, native, prepared, resident) test the initial residual only.
        # Hartree_CG_Tolerance alone (criterion 1) runs there.
        features.append(
            "the Poisson CG test against the right-hand side (Hartree_CG_Criterion: 2) of a cluster"
        )
    return features


def _forces_on_kpoint_pools(problem) -> bool:
    """Whether an MPI run sums its forces on the ranks of the k-point pools.

    That is the scalar k-point SCF (:func:`parsec_python.SCF.kpoints.run_scf_kpoints`,
    whose nonlocal force is a sum over the states: ``KPointJob.extras_by_columns``) of a
    cell sampled beyond the Gamma point, without spin, spinors or a hybrid functional,
    whose forces read the orbitals of one process.
    """

    scf = problem.scf
    return bool(
        problem.periodic_cell is not None
        and problem.has_kpoint_sampling
        and not getattr(scf, "noncollinear", False)
        and not getattr(scf, "self_consistent_spin_orbit", False)
        and not getattr(scf, "spin_polarized", False)
        and not getattr(scf, "is_hybrid", False)
        and not _gamma_point_only(problem)
    )


def _mpi_unsupported_features(translation) -> list[str]:
    """Features of the reference drivers that an MPI run (``--mpi``) cannot carry.

    The MPI-distributed k-point solver (:mod:`.SCF.kpoint_mpi`) holds a block of
    grid rows of every orbital on each rank and gathers no orbitals, and it runs
    the device eigensolver (CHEBFF/CHEBDAV with SUBSPACE steps) only.  What needs
    the whole orbitals or a host eigensolver is refused by name instead of being
    computed from partial rows or an empty block.
    """

    problem = translation.problem
    features = []
    if any(
        getattr(specification, "has_hubbard", False)
        for specification in problem.pseudopotentials.values()
    ):
        features.append(
            "LDA+U (its occupation matrices need the orbitals on every grid point)"
        )
    method = problem.eigensolver.method
    if method in ("arpack", "diagla"):
        features.append(
            f"Eigensolver: {method} (a host eigensolver of the whole basis)"
        )
    uses_kpoint_solver = (
        problem.periodic_cell is not None
        or problem.has_kpoint_sampling
        or problem.scf.noncollinear
    )
    if (
        uses_kpoint_solver
        and (
            getattr(translation, "compute_forces", False)
            or getattr(translation, "relaxation", None) is not None
            or getattr(translation, "molecular_dynamics", None) is not None
        )
        and not _forces_on_kpoint_pools(problem)
    ):
        features.append(
            "forces, relaxation or molecular dynamics of a periodic run at the Gamma point alone, "
            "with spin, a hybrid functional or spinors (they read the orbitals, which the distributed "
            "solver does not gather; those of the scalar k-point SCF are summed on the ranks of its "
            "k-point pools; the forces and relaxations of a bulk cell at the Gamma point run on "
            "several GPUs and nodes in the sectors of its point group with "
            "scripts/run_multi_gpu_scf.sh [--rank-per-gpu])"
        )
    dos = getattr(translation, "dos", None)
    if dos is not None and getattr(dos, "ylm_l", -1) >= 0:
        features.append("the Ylm-projected density of states (YLM_DOS_L)")
    if getattr(translation, "gw", None) is not None:
        features.append("the BerkeleyGW files (Output_GW)")
    if getattr(problem, "embedding", None) is not None:
        # FDET of PARSEC 1.5: one process holds the frozen sample field.
        features.append("FDET (Potential_Field)")
    if getattr(problem.scf, "is_hybrid", False):
        # The rows of every orbital are divided among the ranks here; the
        # hybrid of a cluster runs on several nodes in the MPI runner, whose
        # ranks divide the symmetry sectors (SCF/hybrid_mpi.py).
        features.append(
            f"the self-consistent hybrid functional (Correlation_Type: {problem.scf.xc_functional}; "
            "on several nodes a cluster's hybrid runs with scripts/run_multi_gpu_scf.sh)"
        )
    if str(getattr(problem.scf, "xc_functional", "")).startswith("libxc:") and not (
        getattr(problem.scf, "is_hybrid", False) or getattr(problem.scf, "is_meta_gga", False)
    ):
        # Not tested with the distributed solvers (their XC paths on rows of the grid).
        features.append(f"the libxc functional {problem.scf.xc_functional} (not tested with --mpi)")
    if getattr(problem.scf, "is_meta_gga", False):
        # tau (and g) are sums over the orbitals on every grid point, and the
        # orbital terms of the Hamiltonian reach the neighbours' rows.
        features.append(
            f"the meta-GGA {problem.scf.xc_functional} (its kinetic-energy density needs the orbitals "
            "on every grid point)"
        )
    return features


def _hybrid_on_device(arguments: argparse.Namespace, translation) -> bool:
    """Whether the device Fock loop runs this input's hybrid functional.

    The input has one, its pairs are solved by the Kronecker solver (the
    default ``Exx_Pair_Solver``), the backend resolves to CuPy and the run is
    one process (no ``--mpi``): the GPUs of one node.  A periodic cell
    (bulk or slab) takes this loop at the Gamma point, unpolarized (real
    orbitals, pair densities by cuFFT); with k-points or spin it runs the
    Fock loop of :mod:`parsec_python.SCF.hybrid_periodic` on the reference
    solver, whose k-point workers and exchange use the GPUs
    (:mod:`parsec_python.acceleration.SCF.hybrid_periodic`).  ``False`` for
    every other input, which the reference solver then runs.  (Under
    ``PARSEC_GAMMA_HYBRID_SECTORS=kpoint`` a bulk cell at the Gamma point large
    enough for the k-point sectors then goes to that loop as the 1x1x1 mesh:
    :func:`_gamma_hybrid_sector_handover`.)
    """

    from .Parallel import mpi

    problem = translation.problem
    if not getattr(problem.scf, "is_hybrid", False):
        return False
    if getattr(arguments, "mpi", False) or mpi.requested():
        return False
    if problem.periodic_cell is not None:
        if problem.scf.spin_polarized or (problem.has_kpoint_sampling and not _gamma_point_only(problem)):
            # The Bloch Fock loop on the reference solver (its device k-point pool and exchange).
            return False
    else:
        # The device solves the pair densities of a cluster with the Kronecker solver only;
        # Exx_Pair_Solver: cg runs on the reference solver.
        settings = getattr(problem.scf, "exact_exchange", None)
        if getattr(settings, "pair_solver", "kronecker") != "kronecker":
            return False
    try:
        selection = resolve_backend(arguments.backend, translation.problem)
    except (BackendUnavailableError, NotImplementedError, ValueError):
        return False
    return selection.selected == "cupy"


def _gamma_hybrid_sector_handover(arguments: argparse.Namespace, translation) -> int | None:
    """The exit status of the reference solver for an unpolarized Gamma-point hybrid that the k-point sectors take,
    ``None`` for any other input (which the device Fock loop keeps).

    Asked where the device Fock loop would run the input (:func:`_hybrid_on_device`, nothing else reference-only):
    under ``PARSEC_GAMMA_HYBRID_SECTORS=kpoint`` a bulk cell large enough for the k-point sectors runs as the 1x1x1
    mesh of the reference solver, whose GPU k-point workers solve the sectors of the little group of Gamma
    (:func:`.SCF.gamma_hybrid_route.gamma_hybrid_sector_route`; ``auto``, the default, keeps it in the driver, in the
    real sectors of the Gamma route, and ``off`` on the full grid).  The reference solver takes the same decision
    from the same input.
    """

    from .SCF.gamma_hybrid_route import gamma_hybrid_sector_route

    try:
        route = gamma_hybrid_sector_route(translation)
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if route is None:
        return None
    if getattr(arguments, "mpi_rank", 0) == 0 and not arguments.quiet:
        print(route.note(), file=sys.stderr)
    return reference_main(_reference_argv(arguments))


def _backend_resolves_to_cupy(arguments: argparse.Namespace, translation) -> bool:
    """Whether the backend of this run resolves to CuPy (``False`` where it cannot be resolved)."""

    try:
        selection = resolve_backend(arguments.backend, translation.problem)
    except (BackendUnavailableError, NotImplementedError, ValueError):
        return False
    return selection.selected == "cupy"


def _spin_symmetry_off(arguments: argparse.Namespace, translation) -> bool:
    """Whether the solver of a periodic cell takes no point group by request: ``--symmetry off``, or
    ``Ignore_Symmetry`` without a ``--symmetry`` option (the rule of ``solver_symmetry_mode`` in :func:`_run`)."""

    requested = getattr(arguments, "symmetry", None)
    return requested == "off" or (requested is None and bool(getattr(translation, "ignore_symmetry", False)))


def _gamma_spin_on_device(
    arguments: argparse.Namespace, translation, *, supercell: bool = False, notes: list | None = None
) -> bool:
    """Whether the accelerated Gamma sectors run this spin-polarized periodic input (``PARSEC_GAMMA_SPIN_SECTORS``).

    :func:`parsec_python.acceleration.SCF.gamma_spin.gamma_spin_on_sector_route`: a collinear single point of a bulk
    cell at the Gamma point that the route of the sectors takes, in one process, on a backend that resolves to CuPy,
    and its forces and relaxations by an optimizer where they keep the sectors (``PARSEC_SPIN_SECTOR_FORCES=auto``;
    ``supercell``: the input is the supercell of a k-point mesh, whose forces keep the reference solver), and its
    forces and moves on the full grid where both channels fit this process (``notes`` receives the NOTE of one that
    does not).  ``False`` keeps the reference solver's 1x1x1 k-point driver of before.
    """

    from .Parallel import mpi
    from .SCF.gamma_features import gamma_feature_settings, ldau_spin_full_grid_setting
    from .SCF.gamma_spin import gamma_spin_on_sector_route, gamma_spin_sector_setting, spin_sector_forces_setting

    gamma_spin_sector_setting()
    spin_sector_forces_setting()
    # (TS, a meta-GGA and libxc take this route under their switches: SCF/gamma_features.py)
    gamma_feature_settings()
    ldau_spin_full_grid_setting()

    def backend_is_cupy() -> bool:
        try:
            selection = resolve_backend(arguments.backend, translation.problem)
        except (BackendUnavailableError, NotImplementedError, ValueError):
            return False
        return selection.selected == "cupy"

    return gamma_spin_on_sector_route(
        translation,
        mpi=bool(getattr(arguments, "mpi", False) or mpi.requested()),
        backend_is_cupy=backend_is_cupy,
        supercell=supercell,
        symmetry_off=_spin_symmetry_off(arguments, translation),
        symmetry_on=getattr(arguments, "symmetry", None) == "on",
        notes=notes,
    )


def _device_eigensolver_selected(arguments: argparse.Namespace, translation) -> bool:
    """Whether the backend of this run resolves to the CuPy eigensolvers.

    Asked only for an input whose eigensolver the device runs only there
    (``Eigensolver: arpack``); ``False`` for every other input.
    """

    if translation.problem.eigensolver.method != "arpack":
        return False
    try:
        selection = resolve_backend(arguments.backend, translation.problem)
    except (BackendUnavailableError, NotImplementedError, ValueError):
        return False
    return selection.selected == "cupy"


def _cupy_backend_selected(arguments: argparse.Namespace, translation) -> bool:
    """Whether the backend of this run resolves to CuPy, asked for a periodic cell of the Fortran-derived routines
    only (``False`` for every other input): the accelerated route of a cell on Fortran's general grid
    (:func:`.SCF.periodic_routes.general_grid_reference_only`) was validated on the GPU."""

    problem = translation.problem
    if problem.periodic_cell is None or not getattr(problem.grid, "fortran_routines", False):
        return False
    try:
        selection = resolve_backend(arguments.backend, problem)
    except (BackendUnavailableError, NotImplementedError, ValueError):
        return False
    return selection.selected == "cupy"


def _start_compile_ahead(arguments: argparse.Namespace, resources: Resources, translation) -> bool:
    """Start the compilations ahead of a GPU run (``PARSEC_CUPY_COMPILE_AHEAD``); ``True`` where threads started.

    The setting is read in every run (``ValueError`` for a value it does not know).  Nothing starts in a dry run,
    on the CPU, for a host backend or where CuPy is not installed; the threads themselves leave a process without a
    usable device alone (:func:`.backends.compile_ahead.start`).
    """

    from .backends.compile_ahead import compile_ahead_setting, start

    if compile_ahead_setting() == "off":
        return False
    if arguments.dry_run or resources.device == "cpu" or arguments.backend not in ("auto", "cupy"):
        return False
    if not _cupy_installed():
        return False
    return start(translation.problem)


def _gamma_point_only(problem: SinglePointInput) -> bool:
    """Whether the k-point sampling of the input is the Gamma point alone (``mp 1 1 1``, or a list of one k = 0).

    That is the Gamma-point calculation of the accelerated path, not a k-point sampling.
    """

    from parsec_python.Grid import kpoint_sampling

    sampling = kpoint_sampling(problem)
    return sampling is not None and len(sampling[0]) == 1 and not np.any(sampling[0])


_PERIODIC_SECTOR_FORCES_SWITCH = "PARSEC_PERIODIC_SECTOR_FORCES"


def periodic_sector_forces_setting() -> str:
    """``PARSEC_PERIODIC_SECTOR_FORCES``: ``auto`` (the default) or ``off``.

    ``auto`` lets the forces and the relaxation of a bulk cell at the Gamma point keep the sectors of its point group
    (:func:`periodic_sector_forces_refusal` names what does not); ``off`` solves every forces, relaxation and
    molecular-dynamics run of a cell on the full grid, as before the forces in sectors existed.
    """

    value = os.environ.get(_PERIODIC_SECTOR_FORCES_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_PERIODIC_SECTOR_FORCES_SWITCH} must be auto or off, not {value!r}")
    return value


_PERIODIC_SECTOR_BANDS_SWITCH = "PARSEC_PERIODIC_SECTOR_BANDS"


def periodic_sector_bands_setting() -> str:
    """``PARSEC_PERIODIC_SECTOR_BANDS``: ``auto`` (the default) or ``off``.

    ``auto`` lets the SCF of a bulk cell at the Gamma point that asks for a band structure (``bandstruc``) keep the
    sectors of its point group.  ``calc_bands`` reads only what the SCF leaves on the full grid: the converged
    effective potential (the result of the sectors holds it expanded from the orbits of the grid), the Fermi level
    and the number of states (:func:`.Output.post_scf.band_hand_over`); the bands of every k-point of the path are
    solved on the full grid afresh, as before.  ``off`` solves the whole SCF of such a cell on the full grid, as
    before the hand-over existed.
    """

    value = os.environ.get(_PERIODIC_SECTOR_BANDS_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_PERIODIC_SECTOR_BANDS_SWITCH} must be auto or off, not {value!r}")
    return value


def periodic_sector_forces_refusal(
    translation, *, supercell: bool = False, moving_sectors: bool = False, fortran_fixed_atoms: bool = False
) -> str | None:
    """Why the forces or the moved atoms of a periodic cell keep the full grid, or ``None`` where sectors take them.

    ``moving_sectors``: the caller runs the molecular dynamics or manual movements of the cell in the sectors of the
    group its structures keep, checked at every structure (``PARSEC_MOVING_SECTORS=auto``,
    :mod:`.SCF.moving_sectors`, the command line): they are not refused for what they are.  ``fortran_fixed_atoms``
    (the command line under ``PARSEC_MOVING_SECTORS=auto`` where the basis of the cell reaches the floor of the
    sectors, in one process: :func:`.SCF.moving_sectors.fortran_fixed_atoms_lifted`): a relaxation with atoms fixed
    by ``Move_Flag`` on the Fortran-derived routines keeps the sectors of the subgroup that maps the fixed atoms onto
    fixed atoms, as on the default routines (under ``auto`` where they have 4 operations or more,
    :func:`.SCF.moving_sectors.fortran_fixed_minimum_order`); its ionic loop averages the forces and moved positions
    over the operations of the group of those routines and of the sectors that respect the flags.

    The sectors of the point group of a bulk cell at the Gamma point (``PARSEC_PERIODIC_SYMMETRY``, the driver's
    route) give the forces of the density the SCF used once their nonlocal term and the total are averaged over the
    full group ``G`` of the reduction (:mod:`.Forces.sector_symmetry`).  That holds for a single point with forces and
    for a relaxation by an optimizer (``bfgs``, ``simple``), whose ionic loop finds ``G`` once and averages the forces
    and the moved positions over it (``parsec.F90`` keeps the structure of a relaxation in its group in the same way).
    The Fortran-derived routines (``Use_Fortran_Routines``, also where the parser chose them: Cubic_Spline forces,
    an FHI pseudopotential, a core correction, ``Restart_Run``, ...) do the same under
    ``PARSEC_FORTRAN_SECTOR_FORCES=auto`` (their terms on the fields the sectors expand, their nonlocal term in the
    sectors: :func:`.Forces.periodic_device.compute_device_periodic_forces`); under ``off`` their forces read the full
    grid, as before.
    The rest stays on the full grid: molecular dynamics and manual movements (``parsec.F90`` uses no symmetry there),
    ``Move_Flag`` fixing atoms under ``PARSEC_MOVE_FLAG_SUBGROUP=off`` (a fixed atom that ``G`` exchanges with a
    movable one leaves the group; under ``auto`` a relaxation keeps the sectors of the subgroup that maps the fixed
    atoms onto fixed atoms, which the driver detects with them as a species of their own, and a single point moves no
    atom; the relaxation with fixed atoms of a spin-polarized cell keeps the full grid of both channels,
    ``PARSEC_SPIN_SECTOR_FORCES``, and that of a cell on the Fortran-derived routines the full grid of those
    routines, which symmetrize with their own group), point charges, a band structure under
    ``PARSEC_PERIODIC_SECTOR_BANDS=off`` (under ``auto`` the bands take the expanded potential of the sectors), a
    non-self-consistent run, a restart from ``wfn.dat`` (full-grid states), an electric field
    (``Polarizability``), k-point sampling beyond the Gamma point (its own routes, e.g. the force wedge of the mesh)
    and the supercell that stands for a mesh of the time-reversal invariant points (``supercell``: it repeats the
    atoms of the input).  ``PARSEC_PERIODIC_SECTOR_FORCES=off`` keeps every one on the full grid, as before.
    """

    from .Forces.sector_symmetry import fortran_sector_forces_setting

    problem = translation.problem
    if problem.periodic_cell is None:
        return "not a periodic cell"
    if periodic_sector_forces_setting() == "off":
        return f"{_PERIODIC_SECTOR_FORCES_SWITCH}=off"
    if getattr(translation, "molecular_dynamics", None) is not None and not moving_sectors:
        return "molecular dynamics uses no symmetry"
    relaxation = getattr(translation, "relaxation", None)
    if relaxation is not None and relaxation.method not in ("bfgs", "simple") and not moving_sectors:
        return "manual movements use no symmetry"
    if (
        relaxation is None
        and not getattr(translation, "compute_forces", False)
        and getattr(translation, "molecular_dynamics", None) is None
    ):
        return "no forces are computed"
    if (
        getattr(getattr(translation, "routines", None), "fortran", False)
        or getattr(problem.grid, "fortran_routines", False)
    ) and fortran_sector_forces_setting() == "off":
        # PARSEC_FORTRAN_SECTOR_FORCES=auto: their forces and moves keep the sectors as well.
        return (
            "the forces of the Fortran-derived routines (Use_Fortran_Routines) read the full grid "
            "(PARSEC_FORTRAN_SECTOR_FORCES=off)"
        )
    from .SCF.gamma_features import LDAU_FORCES_SWITCH, ldau_sector_forces_setting

    # (read in every forces run of a cell, so that a value it does not know is refused)
    ldau_forces = ldau_sector_forces_setting()
    if (
        getattr(getattr(translation, "routines", None), "fortran", False)
        or getattr(problem.grid, "fortran_routines", False)
    ) and (
        getattr(getattr(problem, "scf", None), "van_der_waals", False)
        or (
            any(getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values())
            and ldau_forces == "off"
        )
    ):
        # TS and LDA+U take the Gamma sectors for their SCF (PARSEC_GAMMA_TS_SECTORS,
        # PARSEC_GAMMA_LDAU_SECTORS, validated for single points); their forces on the Fortran-derived routines (every
        # TS input) keep the full grid, as before PARSEC_FORTRAN_SECTOR_FORCES, whose sector forces were validated
        # without these corrections.  LDA+U keeps the sectors under PARSEC_LDAU_SECTOR_FORCES=auto (its term of
        # every atom from the sector bases, Hamiltonian/hubbard_device.DeviceHubbard.forces).
        return (
            "the forces of the TS correction or of LDA+U on the Fortran-derived routines read the full grid (their "
            f"sector forces are not validated; LDA+U: {LDAU_FORCES_SWITCH}={ldau_forces})"
            if not getattr(getattr(problem, "scf", None), "van_der_waals", False)
            else "the forces of the TS correction or of LDA+U on the Fortran-derived routines read the full grid (their "
            "sector forces are not validated)"
        )
    movable = getattr(translation, "movable", None)
    if movable is not None and not all(bool(flag) for flag in movable):
        from parsec_python.Relaxation.run import MOVE_FLAG_SUBGROUP_SWITCH, move_flag_subgroup_setting

        # PARSEC_MOVE_FLAG_SUBGROUP=auto: a relaxation by an optimizer keeps the sectors of the group the detection
        # finds with the fixed atoms as a species of their own (every operation maps them onto fixed atoms:
        # parsec_python.Relaxation.run.symmetry_fixed_atoms), and the flags of a single point with forces move no
        # atom; off keeps both on the full grid, as before.
        if move_flag_subgroup_setting() == "off":
            return f"Move_Flag fixes atoms ({MOVE_FLAG_SUBGROUP_SWITCH}=off)"
        if relaxation is not None and getattr(getattr(problem, "scf", None), "spin_polarized", False):
            # The subgroup sectors of Move_Flag are validated for unpolarized cells; a spin-polarized
            # cell relaxed with fixed atoms keeps the full grid of both channels that PARSEC_SPIN_SECTOR_FORCES gives
            # it (SCF/gamma_spin._full_grid_moves, validated against the k-point driver), and the runner refuses it
            # on several ranks.  Its single-point forces move no atom and keep the sectors.
            return (
                "Move_Flag fixes atoms of a spin-polarized cell (the subgroup sectors of "
                f"{MOVE_FLAG_SUBGROUP_SWITCH} are not validated for two channels: both on the full grid)"
            )
        if relaxation is not None and not fortran_fixed_atoms and (
            getattr(getattr(translation, "routines", None), "fortran", False)
            or getattr(problem.grid, "fortran_routines", False)
        ):
            # The forces of the Fortran-derived routines in sectors (PARSEC_FORTRAN_SECTOR_FORCES) are averaged over the
            # group of their own operations (which ignore Move_Flag, as PARSEC 1.5 does) and of the sectors; the ionic
            # loop keeps the operations of that group that respect the flags.  Under PARSEC_MOVING_SECTORS=auto the
            # command line takes such a relaxation of a cell above the floor of the sectors in the sectors of the
            # subgroup of the fixed atoms, where they have 4 operations or more (fortran_fixed_atoms, and the
            # minimum_sector_order of the preparation); otherwise, and under MPI (main.py --mpi refuses the forces of a
            # Gamma cell), it keeps their full grid, as before.  The MPI runner takes it in the sectors of that
            # subgroup whatever their number (PARSEC_RUNNER_FORTRAN_FIXED_SECTORS=auto, since 2026-11-22).
            return (
                "Move_Flag fixes atoms of a cell on the Fortran-derived routines (Use_Fortran_Routines), whose forces "
                f"are symmetrized with their own group (the subgroup sectors of {MOVE_FLAG_SUBGROUP_SWITCH} are "
                "validated on the default routines)"
            )
    if getattr(problem, "point_charges", ()):
        return "point charges (Add_Point_Charges)"
    if getattr(translation, "polarizability_field", None) is not None:
        return "an electric field (Polarizability)"
    if problem.bands is not None and periodic_sector_bands_setting() == "off":
        # Under PARSEC_PERIODIC_SECTOR_BANDS=auto the bands take the expanded potential of the sectors.
        return "a band structure"
    if getattr(translation, "nscf", None) is not None:
        # nscf_kpoints runs no SCF (Restart_Run is required; the reference CLI runs it): never reached from main.
        return "a non-self-consistent run"
    if getattr(translation, "restart_run", False) and getattr(translation, "restart_from_wfndat", False):
        return "Restart_From_Wfndat (the stored orbitals seed the eigensolver of the full grid)"
    if supercell:
        return "the supercell of a k-point mesh, which repeats the atoms of the input"
    if problem.has_kpoint_sampling and not _gamma_point_only(problem):
        return "k-point sampling beyond the Gamma point"
    return None


def _periodic_sector_forces_line(system, operations) -> str | None:
    """The parsec.out line of the forces of a cell in sectors, or ``None`` where the cell is on the full grid.

    With the full group ``G`` of a point-group reduction (``operations``): ``|G|``, the atoms whose projectors the
    sectors keep (one per orbit of the group of the sectors:
    :func:`.Symmetry.axis_reflection.compress_projectors_for_sectors`, ``PARSEC_SECTOR_PROJECTOR_ORBITS``; the
    nonlocal forces contract the gradients of the same columns) and their columns.  The sectors of an Abelian group
    solve every level; they keep the projectors of one atom per orbit where the reduction knows the orbits of its atoms
    (their nonlocal term completed by the average over the group), of every atom otherwise.
    """

    from .Forces.sector_symmetry import sector_reduction
    from .Symmetry.axis_reflection import projector_orbit_counts

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    solved = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if solved is None or int(getattr(solved, "group_order", 1)) <= 1:
        return None
    atom_count = len(system.atoms)
    operator = getattr(system, "nonlocal_operator", None)
    labels = () if operator is None else tuple(getattr(operator, "labels", ()))
    if operations is None:
        # Abelian sectors: the projectors of one atom per orbit where the reduction was given the orbits
        # (Symmetry/axis_reflection.attach_abelian_atom_orbits), every atom otherwise.
        kept_atoms, kept_columns = projector_orbit_counts(solved, labels, atom_count)
        return (
            f" Forces of the periodic cell in the sectors of its point group: an Abelian group of "
            f"{int(solved.group_order)} operations, every level solved, "
            + (
                "the projectors of every atom "
                if kept_columns == len(labels)
                else f"projectors of {kept_atoms} of {atom_count} atoms ({kept_columns} of {len(labels)} columns) "
            )
            + f"({_PERIODIC_SECTOR_FORCES_SWITCH}=auto)"
        )
    reduction = sector_reduction(system)
    kept_atoms, kept_columns = projector_orbit_counts(reduction, labels, atom_count)
    return (
        f" Forces of the periodic cell in the sectors of its point group: averaged over {len(operations)} "
        f"operations; projectors of {kept_atoms} of {atom_count} atoms ({kept_columns} of {len(labels)} columns) "
        f"({_PERIODIC_SECTOR_FORCES_SWITCH}=auto)"
    )


def _fortran_sector_forces_line(system, movable=None) -> str | None:
    """The parsec.out line of the forces of a cell of the Fortran-derived routines in sectors, or ``None``.

    How many operations their total and the moved positions of a relaxation are averaged over: the group that the
    operations of those routines (:func:`parsec_python.Forces.periodic.periodic_force_operations`) and of the sectors
    generate (:func:`.Forces.sector_symmetry.fortran_force_operations`, ``PARSEC_FORTRAN_SECTOR_FORCES=auto``).
    ``movable``: the flags of a relaxation whose ``Move_Flag`` fixes atoms; the ionic loop then averages the total
    force and the moved positions of every structure over the operations of that group, as found for the first
    structure, that map the fixed atoms onto fixed atoms (:func:`parsec_python.Forces.operations_preserving`), and
    the line names how many those are (since 2026-11-22).
    """

    from parsec_python.Forces.periodic import periodic_force_operations
    from parsec_python.Grid.pbc import uses_fortran_routines

    from .Forces.sector_symmetry import fortran_force_operations, sector_symmetry_operations

    if getattr(getattr(system, "input", None), "periodic_cell", None) is None or not uses_fortran_routines(
        system.grid
    ):
        return None
    sectors = sector_symmetry_operations(system)
    if sectors is None:
        return None
    routine = periodic_force_operations(system, None)
    combined = fortran_force_operations(routine, sectors)
    flags = None if movable is None else np.asarray(movable, dtype=bool)
    if flags is not None and flags.size == len(system.atoms) and not flags.all():
        from parsec_python.Forces import operations_preserving

        kept = operations_preserving(combined, flags)
        return (
            " Forces of the Fortran-derived routines in the sectors: the total and the moved positions averaged over "
            f"the {len(kept)} of the {len(combined)} operations (the group of their {len(routine)} and of the "
            f"{len(sectors)} of the sectors) that map the atoms Move_Flag fixes onto fixed atoms, those of the first "
            "structure for every structure (PARSEC_FORTRAN_SECTOR_FORCES=auto)"
        )
    return (
        " Forces of the Fortran-derived routines in the sectors: their own terms, the total and the moved positions "
        f"averaged over {len(combined)} operations (the group of their {len(routine)} and of the {len(sectors)} of "
        "the sectors; PARSEC_FORTRAN_SECTOR_FORCES=auto)"
    )


def _hybrid_full_grid_forces_line(system) -> str | None:
    """The parsec.out line of a Gamma-point hybrid on the full grid below the floor of its sectors whose forces and
    moved positions are averaged over the operations of the cell, or ``None`` for every other run.

    ``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto`` on the default routines (the detail of
    :func:`.SCF.gamma_hybrid_route.full_grid_forces_averaged`): the operations the Fortran-derived routines take on the
    full grid (:func:`parsec_python.Forces.periodic.periodic_force_operations`), those of them that respect the
    movable flags in a relaxation (the ionic loop keeps them).
    """

    from parsec_python.Forces.periodic import periodic_force_operations

    from .SCF.gamma_hybrid_route import full_grid_forces_averaged

    if not full_grid_forces_averaged(system):
        return None
    operations = periodic_force_operations(system, None)
    return (
        " Forces of the periodic hybrid on the full grid (below the floor of its sectors): the total and the moved "
        f"positions averaged over the {len(operations)} operations of the cell (PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto)"
    )


def _supercell_kpoints_handover(translation) -> bool:
    """Whether ``PARSEC_SUPERCELL_KPOINTS=auto`` solves this Gamma-point cell as a smaller cell with k-points, which
    the reference CLI runs (``parsec_python/SCF/supercell_mesh.py``; it makes the same cell and mesh).  A switch
    value it refuses is handed over too: the reference CLI reports it as an input error.  Where the cell repeats a
    smaller one that cannot take its grid (too small for the stencil), the note says why it stays here, at the Gamma
    point."""

    from parsec_python.SCF.supercell_mesh import supercell_mesh_route

    notes: list[str] = []
    try:
        found = supercell_mesh_route(translation, notes=notes)
    except ValueError:
        return True
    for note in notes:
        print(f"NOTE: {note}")
    return found is not None


def _reference_argv(arguments: argparse.Namespace) -> list[str]:
    """Forward the options the reference CLI shares with this one."""

    argv = [str(arguments.input)]
    for flag, value in (
        ("--pp-dir", arguments.pp_dir),
        ("--output", arguments.output),
        ("--log", arguments.log),
        ("--bands-from", arguments.bands_from),
        ("--kpoint-workers", arguments.kpoint_workers),
        ("--kpoint-pools", getattr(arguments, "kpoint_pools", None)),
        ("--restart-from", arguments.restart_from),
        ("--energy-refinement", arguments.energy_refinement),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if arguments.mlff is not None:
        argv += ["--mlff"] + ([arguments.mlff] if arguments.mlff else [])
    for flag, enabled in (
        ("--no-archive", arguments.no_archive),
        ("--save-wavefunctions", arguments.save_wavefunctions),
        ("--dry-run", arguments.dry_run),
        ("--quiet", arguments.quiet),
        ("--debug", arguments.debug),
        ("--esdf-lenient", getattr(arguments, "esdf_lenient", False)),
        # the ranks other than 0 of an MPI run (_join_mpi)
        ("--no-data-files", getattr(arguments, "no_data_files", False)),
    ):
        if enabled:
            argv.append(flag)
    return argv


def _join_mpi(arguments: argparse.Namespace) -> int | None:
    """Make this process one rank of an MPI run; ``None`` if all is well, else an exit status.

    Every rank runs the whole calculation (see :mod:`.Parallel.mpi`); the ranks other than 0 are
    silent and write no files.
    """
    from .Parallel import mpi

    if not (arguments.mpi or mpi.requested()):
        return None
    os.environ[mpi.MPI_ENV] = "1"
    try:
        context = mpi.context()
    except ImportError as error:
        print(f"Input error: --mpi needs mpi4py: {error}", file=sys.stderr)
        return 2
    if arguments.dry_run:
        return None
    arguments.mpi_rank = context.rank
    if not context.is_root:
        arguments.quiet = True
        arguments.log = Path(os.devnull)
        arguments.no_archive = True
        arguments.output = None
        arguments.no_data_files = True  # parsec.dat, eigen.dat, cube files, qpoints.dat: rank 0 writes them
    elif context.size > 1:
        print(
            f"MPI: {context.size} ranks, {context.ranks_per_node} per node; the k-points are solved by "
            "pools of ranks (spin-orbit and non-collinear bases split by blocks of grid rows)",
            flush=True,
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    status = _join_mpi(arguments)
    if status is not None:
        return status
    try:
        resources = resolve_resources(
            device=arguments.device,
            gpus=arguments.gpus,
            gpu_ids=arguments.gpu_ids,
            threads=arguments.threads,
            backend=arguments.backend,
        )
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    # The request reaches the solver through its variables, for this
    # calculation only: a process that runs several keeps none of them.
    with resources.applied(), ExitStack() as scope:
        try:
            status = _run(arguments, resources, scope)
        except BaseException as error:
            _abort_mpi(1, error)
            raise
    if status not in (0, 3, 4):
        # 3 (an SCF loop did not converge) and 4 (a relaxation not relaxed: Movement_Num used up) are outcomes of
        # the run that every rank reaches together (relaxation_exit_code of the reference CLI): no abort for them
        _abort_mpi(status)
    return status


def shared_vector_seed(translation, context=None):
    """The translation with rank 0's start-vector seed on every MPI rank.

    Every rank parses the input itself, and ``Vector_Rng_Seed: -1`` takes the seed from the clock during
    the parse (``Input/solver_options.py``), so ranks whose parse falls into different seconds would draw
    different seeds.  Each rank builds its own columns of the start basis from its seed
    (``Eigensolvers/mpi_shared_basis.py``): the basis would mix streams, and the seed rank 0 prints and
    writes to bak_complete_parsec.in would not repeat the run.
    PARSEC 1.5 reads the input on one process.  An explicit seed is the same on every rank and stays.
    ``context`` is the MPI context (:func:`.Parallel.mpi.context` when ``None``); every rank of a run with
    more than one calls this function at the same place (one broadcast).
    """
    from .Parallel import mpi

    if context is None:
        context = mpi.context()
    if context is None or context.size == 1:
        return translation
    eigensolver = translation.problem.eigensolver
    own = (eigensolver.vector_seed, eigensolver.random_seed)
    seeds = tuple(context.bcast(own, root=0))
    if seeds == own:
        return translation
    return replace(
        translation,
        problem=replace(
            translation.problem,
            eigensolver=replace(eigensolver, vector_seed=seeds[0], random_seed=seeds[1]),
        ),
    )


def _abort_mpi(status: int, error: BaseException | None = None) -> None:
    """A rank that failed must not leave the others waiting in a collective (the interpreter's exit
    would wait for them in MPI_Finalize): end the whole MPI job.

    The traceback of ``error`` goes to the standard error first, with the rank: the abort ends the
    process before the interpreter would print it, and the ranks other than 0 write no log.
    """
    from .Parallel import mpi

    context = mpi.context()
    if context is not None and context.size > 1:
        if error is not None and not isinstance(error, SystemExit):
            import traceback

            text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            print(f"MPI rank {context.rank} failed:\n{text}", file=sys.stderr, end="")
        sys.stdout.flush()
        sys.stderr.flush()
        context.comm.Abort(int(status))


def _apply_restarts(translation, arguments):
    """``(translation, relax_restart, dynamics_restart, notes)``: ``Relax_Restart`` or ``--restart-from``, then the
    restart of a dynamics run, applied to ``translation``."""

    translation, relax_restart, restart_notes = restart_structure(
        translation,
        translation.source.parent,
        None
        if arguments.restart_from is None
        else _resolve_output(arguments.restart_from, translation.source.parent, ""),
    )
    translation, dynamics_restart, dynamics_notes = dynamics_restart_structure(
        translation, translation.source.parent
    )
    return translation, relax_restart, dynamics_restart, [*restart_notes, *dynamics_notes]


def _found_problem(found_supercell, problem_of_input):
    """The problem the restarts started from: the supercell where the route took it, else that of the input."""

    return problem_of_input if found_supercell is None else found_supercell[0]


def _run(
    arguments: argparse.Namespace, resources: Resources, scope: ExitStack | None = None
) -> int:
    if arguments.profile_repeats < 1:
        print("Input error: --profile-repeats must be positive", file=sys.stderr)
        return 2
    if arguments.bands_from is not None:
        # Bands from a stored SCF potential only exist in the reference solver.
        return reference_main(_reference_argv(arguments))
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
            esdf_lenient=getattr(arguments, "esdf_lenient", False),
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    try:
        # The switches of the local forces of a cell are used after the first
        # SCF, also by the k-point runs of the reference solver: read here so
        # that a value they do not know stops every run before it.
        from .Forces.periodic_local import validate_local_force_settings

        validate_local_force_settings()
        # PARSEC_SUPERCELL_KPOINTS_MESH and PARSEC_SUPERCELL_KPOINTS_WEDGE (SCF/supercell_mesh.py), read in every run
        # so that a value they do not know is refused before anything is prepared
        from parsec_python.SCF.supercell_mesh import validate_settings as _validate_supercell_mesh

        _validate_supercell_mesh()
        # PARSEC_CUPY_COMPILE_AHEAD (backends/compile_ahead.py), read in every run so that a value it does not know
        # is refused; a GPU run starts compiling the kernels its input will need, beside everything below
        _start_compile_ahead(arguments, resources, translation)
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    translation = shared_vector_seed(translation)
    if arguments.mlff is not None:
        from parsec_python.MLFF import apply_mlff_option

        translation = apply_mlff_option(translation, arguments.mlff)
    if arguments.energy_refinement is not None:
        from parsec_python.Energy.refinement import apply_refinement_option

        try:
            translation = apply_refinement_option(
                translation, arguments.energy_refinement
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
    if translation.problem.periodic_cell is not None:
        # The switches of the k-point sectors and of the detection of their cell group
        # (PARSEC_SYMMETRY_ATOM_TOLERANCE, PARSEC_PERIODIC_DEFECT_CENTRE, ...), read before any hand-over to the
        # reference solver, so that a value they do not know is refused here and not at the setup of its k-point
        # solver (the driver reads them for the routes of this path).
        from .SCF.kpoint_sectors import check_kpoint_sector_switches

        try:
            check_kpoint_sector_switches(translation.problem)
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
    # A k-point mesh of the time-reversal invariant points as the Gamma point of the supercell
    # (PARSEC_KPOINT_SUPERCELL: auto, the default since 2026-11-19, for a single point whose estimated difference of
    # the two kinetic forms is small; on wherever it applies), and a repeating Gamma-point cell as its smaller cell
    # with a mesh (PARSEC_SUPERCELL_KPOINTS=auto, a k-point run of the reference CLI, which makes the same cell):
    # parsec_python/SCF/kpoint_supercell.py, the same energies with the Bloch kinetic term.
    from parsec_python.SCF.kpoint_supercell import (
        cell_parsec_dat_view as _cell_parsec_dat_view,
        cell_result_view as _cell_result_view,
        kpoint_supercell_setting as _supercell_setting,
        kpoints_of_supercell as _bloch_cells,
        per_cell_lines as _per_cell_lines,
        process_gpu_count as _process_gpu_count,
        trim_supercell_route as _trim_supercell_route,
        validate_settings as _validate_supercell_settings,
    )

    from .driver import periodic_sector_refusal as _sector_refusal
    from .Parallel import mpi as _mpi_of_route

    problem_of_input = translation.problem
    translation_of_input = translation
    restarts = None
    try:
        _validate_supercell_settings()
        if _supercell_setting() == "auto":
            # Relax_Restart, --restart-from and the restart of a dynamics run before the route of a mesh: their
            # structure stands for the atoms of the input, of which a supercell is made (on and off keep the order
            # of before below; on's relaxations write the files of the supercell)
            restarts = _apply_restarts(translation, arguments)
            translation = restarts[0]
        _trim_route = _trim_supercell_route(
            translation,
            refusal=_sector_refusal,
            device=resources.device,
            # --mpi: the ranks divide the k-points
            kpoint_ranks=bool(getattr(arguments, "mpi", False) or _mpi_of_route.requested()),
            # the GPUs of this process: the k-point pools divide the k-points over several
            gpus=_process_gpu_count,
            orbitals=bool(getattr(arguments, "save_wavefunctions", False)),
        )
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    _mesh_problem = translation.problem
    _found_supercell = (_trim_route.problem, _trim_route.repeats) if _trim_route.taken else None
    if _trim_route.note is not None and getattr(arguments, "mpi_rank", 0) == 0:
        print(_trim_route.note)
    if _found_supercell is not None:
        translation = replace(translation, problem=_found_supercell[0])
    elif _supercell_kpoints_handover(translation_of_input):
        # the reference command line decides on the structure of the input, before its restarts (the
        # route of PARSEC_SUPERCELL_KPOINTS is asked only where no TRIM supercell was taken; a k-point input, which
        # TRIM takes, never repeats into a smaller cell there, so the two routes never hand one run back and forth)
        return reference_main(_reference_argv(arguments))

    from parsec_python.MLFF import MLFFError, relax_initial_structure

    try:
        if restarts is None:
            restarts = _apply_restarts(translation, arguments)
            translation = restarts[0]
            restarted = translation.problem is not _found_problem(_found_supercell, problem_of_input)
        else:
            restarted = restarts[0].problem is not problem_of_input
        _, relax_restart, dynamics_restart, restart_notes = restarts
        # Initial_Structure: mlff, unless a restart file gave the structure;
        # a dry run only reports the settings.
        mlff_notes = []
        if not arguments.dry_run:
            translation, mlff_notes = relax_initial_structure(
                translation,
                translation.source.parent,
                restarted=restarted,
                progress=None if arguments.quiet else print,
            )
        restart_notes = [*restart_notes, *mlff_notes]
        # PARSEC_SYMMETRIZE_POSITIONS=on: the positions averaged over the group found within
        # PARSEC_SYMMETRY_ATOM_TOLERANCE, before anything reads them (off, the default: as given).
        translation, position_notes = symmetrize_input_positions(translation)
        restart_notes = [*restart_notes, *position_notes]
        # PARSEC_MOVE_FLAG_SUBGROUP and PARSEC_SYMMETRIZED_DENSITY_SECTORS, read in every run so that a value they
        # do not know is refused.
        move_flag_subgroup_setting()
        from parsec_python.Relaxation.domove import bfgs_first_step_setting

        bfgs_first_step_setting()  # PARSEC_BFGS_FIRST_STEP, refused here too
        from .SCF.density_symmetry import symmetrized_density_sectors_setting

        symmetrized_density_sectors_setting()
        from .Symmetry.axis_reflection import slab_wire_sectors_setting
        from .Symmetry.oblique_frame import oblique_sectors_setting

        # PARSEC_SLAB_WIRE_SECTORS, PARSEC_OBLIQUE_SECTORS, PARSEC_OBLIQUE_FULL_GROUP and
        # PARSEC_GENERAL_GRID_ACCELERATED: refused here too, before an input the reference solver runs (a cell on the
        # general grid of the Fortran-derived routines under PARSEC_GENERAL_GRID_ACCELERATED=off) is handed over without
        # reading them
        from .SCF.periodic_routes import general_grid_accelerated_setting
        from .Symmetry.oblique_frame import oblique_full_group_setting

        slab_wire_sectors_setting()
        oblique_sectors_setting()
        oblique_full_group_setting()
        general_grid_accelerated_setting()
        if _found_supercell is not None:
            # the mesh solved as its supercell: said in parsec.out too
            restart_notes = [*restart_notes, f" {_trim_route.note}"]
        use_symmetry, symmetry_note = symmetry_decision(translation)
        # Whether the atoms stay in the point group of the first structure,
        # which Ignore_Symmetry does not tell.  The command line may overrule
        # that keyword, but not a run that leaves the group.
        keeps_group, group_note = symmetry_decision(
            replace(translation, ignore_symmetry=False)
        )
        # The atoms Move_Flag fixes in a relaxation that keeps the subgroup respecting the flags: a species of their
        # own for the symmetry detection of the driver (PARSEC_MOVE_FLAG_SUBGROUP=auto), None otherwise.
        fixed_atoms = symmetry_fixed_atoms(translation)
    except MLFFError as error:
        print(f"MLFF error: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    # A cell keeps no group in the parser (its forces are not symmetrized):
    # whether its SCF may take the point group of the cell is decided below
    # and by the driver.
    periodic_cell = translation.problem.periodic_cell is not None
    if arguments.symmetry == "on" and not keeps_group and not periodic_cell:
        print(
            "Input error: --symmetry on cannot be used here: "
            + (
                group_note
                or "manual movements and molecular dynamics use no symmetry"
            ),
            file=sys.stderr,
        )
        return 2
    symmetry_mode = (
        arguments.symmetry
        if arguments.symmetry is not None
        and (keeps_group or arguments.symmetry == "off")
        else ("auto" if use_symmetry else "off")
    )
    # Forces are symmetrized where the calculation itself uses the group.
    symmetrize_forces = use_symmetry and symmetry_mode != "off"

    accelerated_self_consistent_soc = False
    if _accelerated_self_consistent_soc_eligible(translation):
        try:
            selection_probe = resolve_backend(arguments.backend, translation.problem)
        except (BackendUnavailableError, NotImplementedError):
            selection_probe = None
        accelerated_self_consistent_soc = (
            selection_probe is not None and selection_probe.selected == "cupy"
        )

    if scope is not None:
        # Forces, relaxation and dynamics of a periodic cell: its ionic fields
        # and their forces on the device (Forces/periodic_device.py), on the
        # path below and on the reference solver alike.
        from .Forces.periodic_device import periodic_ionic_scope

        scope.enter_context(periodic_ionic_scope(translation))
    hybrid_on_device = _hybrid_on_device(arguments, translation)
    soc_sectors = "off"
    try:
        # PARSEC_GAMMA_SPIN_SECTORS and PARSEC_SPIN_SECTOR_FORCES, read in every run so that a value they do not
        # know is refused.
        gamma_spin_notes: list[str] = []
        gamma_spin_on_device = _gamma_spin_on_device(
            arguments, translation, supercell=_found_supercell is not None, notes=gamma_spin_notes
        )
        if accelerated_self_consistent_soc:
            # PARSEC_SOC_SECTORS and PARSEC_SOC_AUTO_FOURFOLD: a value they do not know is refused before the SCF.
            from .SCF.soc_sectors import soc_auto_fourfold_setting, soc_sectors_setting

            soc_sectors = soc_sectors_setting()
            soc_auto_fourfold_setting()
        # PARSEC_ORDERED_SUMS and PARSEC_LONG_ROW_SPLIT (the device products of the k-point, band and spin-orbit
        # operators) and
        # PARSEC_MPI_ACCELERATED_BANDS (bandstruc under --mpi), on every route
        from .Hamiltonian.complex_csr_device import long_row_split_setting, ordered_sums_setting
        from .Output.post_scf import mpi_accelerated_bands_setting

        ordered_sums_setting()
        long_row_split_setting()
        mpi_accelerated_bands_setting()
        # PARSEC_SOC_DEVICE_FORCES and PARSEC_SPHERE_TIES, read in every run so that a value they do not know is refused
        from parsec_python.Grid.cluster import sphere_ties_setting

        from .Forces.spinor_device import soc_device_forces_setting

        soc_device_forces_setting()
        sphere_ties_setting()
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    for note in gamma_spin_notes:
        # (a spin cell whose full grid fits nowhere here: it keeps the k-point driver of the reference solver)
        print(f"NOTE: {note}", file=sys.stderr)
    unsupported = _reference_only_features(
        translation,
        accelerated_backend_selected=accelerated_self_consistent_soc,
        device_eigensolver_selected=_device_eigensolver_selected(arguments, translation),
        hybrid_on_device=hybrid_on_device,
        gamma_spin_on_device=gamma_spin_on_device,
        general_grid_on_device=_cupy_backend_selected(arguments, translation),
    )
    if unsupported:
        features = ", ".join(unsupported)
        if arguments.backend in ("cupy", "native"):
            print(
                f"Input error: {features} is not implemented on the accelerated "
                f"path, so --backend {arguments.backend} cannot run this input; "
                "use --backend scipy (or auto) to run it with the reference "
                "solver",
                file=sys.stderr,
            )
            return 2
        problem = translation.problem
        reference_uses_gpu = (
            problem.periodic_cell is not None
            or problem.has_kpoint_sampling
            or problem.bands is not None
            # the non-collinear driver hands its eigenproblem to the k-point device workers, also
            # for an isolated system
            or problem.scf.noncollinear
        )
        if resources.device == "gpu" and not reference_uses_gpu:
            # Only the k-point drivers, band paths and the non-collinear driver of the reference
            # solver have a GPU path; anything else would run on the CPU under a request for the GPU.
            print(
                f"Input error: {features} cannot run on the GPU here (the accelerated "
                "path does not cover it, or CuPy is not usable), and the reference "
                "solver it would fall back to uses no GPU; use --device cpu or auto",
                file=sys.stderr,
            )
            return 2
        from .Parallel import mpi

        blocked = _mpi_unsupported_features(translation) if mpi.requested() else []
        if blocked:
            print(
                "Input error: "
                + ", ".join(blocked)
                + " is not implemented with the MPI-distributed k-point solver (--mpi); "
                "run without --mpi",
                file=sys.stderr,
            )
            return 2
        if getattr(arguments, "mpi_rank", 0) == 0:  # the other ranks of an MPI run are quiet
            print(
                f"NOTE: {features} is not implemented on the accelerated path; "
                "running the reference parsec_python.cli solver instead"
                + (
                    " (its k-point eigensolvers, the exchange of a periodic hybrid and the force sums of a periodic "
                    "cell use the GPUs when CuPy and a device are available; the output names the device)"
                    if reference_uses_gpu and resources.device != "cpu"
                    else ""
                ),
                file=sys.stderr,
            )
        return reference_main(_reference_argv(arguments))
    if hybrid_on_device and translation.problem.periodic_cell is not None:
        # PARSEC_GAMMA_HYBRID_SECTORS=kpoint: an unpolarized Gamma-point hybrid of a large bulk cell as the 1x1x1
        # k-point mesh of the reference solver, whose k-point sectors split the Gamma point (SCF/gamma_hybrid_route.py).
        status = _gamma_hybrid_sector_handover(arguments, translation)
        if status is not None:
            return status
    if translation.problem.periodic_cell is not None:
        from .Parallel import mpi

        # The forces of a periodic cell read the orbitals of one process:
        # refused under --mpi on this path as on the reference solver.
        blocked = _mpi_unsupported_features(translation) if mpi.requested() else []
        if blocked:
            print(
                "Input error: "
                + ", ".join(blocked)
                + " is not implemented with the MPI-distributed k-point solver (--mpi); "
                "run without --mpi",
                file=sys.stderr,
            )
            return 2

    if accelerated_self_consistent_soc and getattr(translation, "energy_refinement", None) is not None:
        print(
            "Input error: Energy_Refinement is not implemented together with self-consistent spin-orbit coupling",
            file=sys.stderr,
        )
        return 2
    # The report of a self-consistent SOC run that allows symmetry prints PARSEC 1.5's block of the point group
    # (as the Fortran does for spinors), not the Ignore_Symmetry lines of a run without symmetry.
    report_symmetry_mode = symmetry_mode
    # Whether the spinor loop may use the double-group sectors of PARSEC_SOC_SECTORS.
    soc_symmetry = False
    # The forces of a spinor loop on the device (PARSEC_SOC_DEVICE_FORCES) are averaged over the point group of the
    # atoms as those of the reference solver are (its use_symmetry), unless --symmetry off: the choice before the
    # spinor loop prepares the solver without symmetry.
    soc_force_symmetry = symmetrize_forces
    if accelerated_self_consistent_soc:
        try:
            symmetry_mode, soc_symmetry, report_symmetry_mode = _soc_symmetry_choice(
                symmetry_mode,
                arguments.symmetry,
                translation.ignore_symmetry,
                soc_sectors,
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
    symmetrize_forces = symmetrize_forces and symmetry_mode != "off"
    if (
        getattr(translation, "polarizability_field", None) is not None
        and getattr(translation, "energy_refinement", None) is not None
    ):
        print(
            "Input error: Energy_Refinement is not implemented together with "
            "Polarizability",
            file=sys.stderr,
        )
        return 2
    # The symmetry the solver is prepared with; the report keeps the mode above.
    solver_symmetry_mode = symmetry_mode
    # Inputs whose symmetry PARSEC 1.5 switches off, and restarts on the full grid, that keep the sectors of the
    # group they leave intact (SCF/symmetry_off_inputs.py; read here so that a value they do not know is refused).
    from .SCF.symmetry_off_inputs import (
        kept_group_note,
        kept_group_reasons,
        kept_group_warnings,
        read_settings as symmetry_off_settings,
        restart_sector_switch,
    )

    try:
        symmetry_off_settings()
        kept_reasons = kept_group_reasons(translation)
        restart_sectors = restart_sector_switch(translation)
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if (
        kept_reasons
        and arguments.symmetry != "off"
        # the spinor driver takes no sectors (_resolve_symmetry_mode)
        and not accelerated_self_consistent_soc
        and solver_symmetry_mode == "off"
    ):
        solver_symmetry_mode = "auto"
    else:
        # --symmetry off, or another rule of this CLI keeps the full grid: no group is kept (no field loop in sectors)
        kept_reasons = ()
    if (
        translation.restart_run
        and getattr(translation, "restart_from_wfndat", False)
        and restart_sectors is None
    ):
        # The orbitals of wfn.dat are full-grid states of one representation:
        # they seed the eigensolver of the full grid (SCF.restart).  The
        # forces keep the symmetrization of the input.  (A cluster under
        # PARSEC_WFNDAT_RESTART_SECTORS=auto keeps its sectors where the stored
        # density keeps the group: SCF/symmetry_off_inputs.restart_in_sectors.)
        if arguments.symmetry == "on":
            print(
                "Input error: --symmetry on cannot be used with Restart_From_Wfndat: "
                "the stored orbitals seed the eigensolver of the full grid",
                file=sys.stderr,
            )
            return 2
        solver_symmetry_mode = "off"
    scf_settings = translation.problem.scf
    if (
        translation.restart_run
        and scf_settings.spin_polarized
        and scf_settings.spin_symmetry_sector_mode == "auto"
        and restart_sectors is None
    ):
        # parsec.dat holds full-grid fields of a run that may have broken the
        # point group of the atoms, which the sectors would average: the two
        # channels of a restart are solved on the full grid.  (A cluster under
        # PARSEC_SPIN_RESTART_SECTORS=auto keeps the sectors where both stored
        # densities keep the group: SCF/symmetry_off_inputs.restart_in_sectors.)
        translation = replace(
            translation,
            problem=replace(
                translation.problem, scf=replace(scf_settings, spin_symmetry_sectors="off")
            ),
        )
        restart_notes = [
            *restart_notes,
            "NOTE: Restart_Run of a spin-polarized calculation: both channels on the full grid "
            "(Spin_Symmetry_Sectors: off for the restart)",
        ]
    if kept_reasons and (
        solver_symmetry_mode == "off"
        or (scf_settings.spin_polarized and translation.problem.scf.spin_symmetry_sector_mode == "off")
    ):
        # A restart above keeps the full grid (wfn.dat, or both spin channels): so does the input, as before.
        solver_symmetry_mode = "off"
        kept_reasons = ()
    if kept_reasons:
        restart_notes = [*restart_notes, kept_group_note(kept_reasons)]
    # PARSEC_MOVING_SECTORS=auto: molecular dynamics and manual movements in the sectors of the group their
    # structures keep, checked at every structure (SCF/moving_sectors.py); read in every run so that a value it does
    # not know is refused.
    from .SCF.moving_sectors import (
        fortran_fixed_atoms_lifted,
        fortran_fixed_minimum_order,
        moving_sectors_setting,
        plan_moving_sectors,
    )

    moving_plan, moving_reason = None, None
    try:
        moving_wanted = moving_sectors_setting() != "off"
        from parsec_python.Relaxation.run import manual_structures

        from .Parallel import mpi as _moving_mpi

        moving_mpi = _moving_mpi.requested() or bool(getattr(arguments, "mpi", False))
        # a relaxation of a cell on the Fortran-derived routines with atoms fixed by Move_Flag: in the sectors of the
        # subgroup of the fixed atoms where its basis reaches the floor of the sectors (on: whatever the size)
        lift_fortran_fixed_atoms = translation.problem.periodic_cell is not None and fortran_fixed_atoms_lifted(
            translation, mpi_requested=moving_mpi, symmetry_argument=arguments.symmetry
        )
        # (and under auto in sectors of 4 operations or more: fewer keep the full grid of those routines)
        fortran_minimum_order = fortran_fixed_minimum_order(
            translation, lifted=lift_fortran_fixed_atoms, symmetry_argument=arguments.symmetry
        )
        manual_list = None
        if (
            moving_wanted
            and translation.relaxation is not None
            and translation.relaxation.method == "manual"
        ):
            try:
                manual_list = manual_structures(translation, translation.source.parent)
            except (OSError, ValueError):
                # (reported where the movements read manual.dat, as before: the plan takes no structure)
                manual_list = None
        moving_plan, moving_reason = plan_moving_sectors(
            translation,
            dynamics_restart=dynamics_restart,
            manual=manual_list,
            mpi_requested=moving_mpi,
            symmetry_argument=arguments.symmetry,
        )
        if (
            moving_plan is not None
            and translation.relaxation is not None
            and translation.relaxation.max_movements > 0
            and manual_list is None
        ):
            # manual.dat could not be read: the full grid, which reports the error where it reads the file
            moving_plan, moving_reason = None, "manual.dat could not be read: the full grid"
        if moving_plan is not None and not moving_plan.periodic and (
            solver_symmetry_mode != "off" or accelerated_self_consistent_soc
        ):
            moving_plan, moving_reason = None, None
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if moving_plan is not None and not moving_plan.periodic:
        # a cluster: the sectors of the group the plan keeps (with the pseudo-atoms of the state or of the
        # structures of manual.dat, at the preparation below)
        solver_symmetry_mode = "auto"
    from .SCF.moving_sectors import ANTIFERROMAGNETIC_REASON, SWITCH as _MOVING_SWITCH

    if moving_plan is None and moving_reason is not None and moving_reason.startswith(ANTIFERROMAGNETIC_REASON):
        # PARSEC_MOVING_SPIN_SECTORS=auto keeps an antiferromagnetic start on the full grid: said in parsec.out
        restart_notes = [*restart_notes, f"NOTE: {_MOVING_SWITCH}: {moving_reason}"]
    # PARSEC_PERIODIC_SECTOR_FORCES, PARSEC_PERIODIC_SECTOR_BANDS and
    # PARSEC_FORTRAN_SECTOR_FORCES, read in every run so that a value they do
    # not know is refused.
    try:
        periodic_sector_forces_setting()
        periodic_sector_bands_setting()
        from .Forces.sector_symmetry import fortran_sector_forces_setting

        fortran_sector_forces_setting()
        # and the switches of TS, LDA+U, meta-GGA and libxc in the Gamma sectors (SCF/gamma_features.py)
        from .SCF.gamma_features import gamma_feature_settings

        gamma_feature_settings()
        from .SCF.band_sectors import band_kpoint_sectors_setting

        band_kpoint_sectors_setting()
        # and PARSEC_GAMMA_HYBRID_SECTOR_FORCES (a small Gamma hybrid with forces averages them on the full grid, or
        # takes the sectors: driver._gamma_hybrid_full_grid_forces, driver._gamma_hybrid_in_sectors)
        from .SCF.gamma_hybrid_route import gamma_hybrid_sector_forces_setting

        gamma_hybrid_sector_forces_setting()
        # PARSEC_STATES_CLOSE_MULTIPLET: more states where States_Num cuts an occupied multiplet (full-grid loops)
        from parsec_python.SCF.state_extension import close_multiplet_setting

        close_multiplet_setting()
        # PARSEC_EXCHANGE_OCCUPATION_CUT: the occupation an orbital needs to enter the exact exchange
        from parsec_python.V_xc.exact_exchange import exchange_occupation_cut_setting

        exchange_occupation_cut_setting()
        # PARSEC_FOCK_DIIS_TAIL: the orbitals the window of the outer DIIS of a Fock loop stores
        from parsec_python.SCF.fock_diis import fock_diis_tail_setting

        fock_diis_tail_setting()
        # PARSEC_FERMI_GAP_ROOT: the Fermi level of a gap
        from parsec_python.Occupations.fermi_dirac import fermi_gap_root_setting

        fermi_gap_root_setting()
        # PARSEC_FILTER_DEGREE_STALL: the starting later-SCF degree back where a loop stalls after lowering it
        from parsec_python.SCF.filter_degree import filter_degree_stall_setting

        filter_degree_stall_setting()
        # PARSEC_KPOINT_WAVEFUNCTIONS and PARSEC_GAMMA_PARSEC_DAT (parsec.dat of the k-point loops)
        from parsec_python.Output.parsec_dat import validate_settings as validate_parsec_dat_settings

        validate_parsec_dat_settings()
        # PARSEC_LDAU_SMOOTH_PROJECTORS: the shape of the LDA+U projectors (Hamiltonian/hubbard.py)
        from parsec_python.Hamiltonian.hubbard import ldau_smooth_projectors_fraction

        ldau_smooth_projectors_fraction()
        # PARSEC_GGA_NUCLEAR_GRADIENT: the density gradient of the gradient functionals beside the nuclei
        from parsec_python.V_xc.nuclear_gradient import forces_refusal, nuclear_gradient_setting

        nuclear_gradient_setting()
        # PARSEC_NONCOLLINEAR_GGA_FRAME: the frame of a non-collinear gradient functional (V_xc/noncollinear.py)
        from parsec_python.V_xc.noncollinear import noncollinear_gga_frame_setting

        noncollinear_gga_frame_setting()
        refusal = forces_refusal(translation)
        if refusal is not None:
            raise ValueError(refusal)
        if translation.problem.scf.spin_polarized:
            # PARSEC_SPIN_FALLBACK_DENSITY: the density of the full grid a spin cluster leaves its sectors for
            from .SCF.fallback_density import fallback_density_setting

            fallback_density_setting()
            # PARSEC_SPIN_START: the start of the spin loops of the reference solver (a spin cell on its k-point loop)
            from parsec_python.SCF.spin_polarized import spin_start_setting
            from parsec_python.cli import spin_ionic_kpoint_setting

            spin_start_setting()
            spin_ionic_kpoint_setting()
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    # Whether the forces or the relaxation of a periodic cell keep the sectors
    # of its point group (periodic_sector_forces_refusal).
    periodic_sector_forces = False
    if translation.problem.periodic_cell is not None:
        # The parser keeps no point group for a cell on the default routines
        # (use_symmetry: its forces are not symmetrized).  The SCF of a bulk
        # cell at the Gamma point may still solve in the sectors of its point
        # group about the origin of the cell, which the driver looks for
        # (PARSEC_PERIODIC_SYMMETRY).  Its single-point forces and its
        # relaxations by an optimizer keep those sectors too, their forces and
        # moved positions averaged over the full group of the sectors
        # (PARSEC_PERIODIC_SECTOR_FORCES=auto), on the Fortran-derived routines
        # as well under PARSEC_FORTRAN_SECTOR_FORCES=auto; molecular dynamics,
        # manual movements and Move_Flag with fixed atoms under
        # PARSEC_MOVE_FLAG_SUBGROUP=off (under auto the sectors of the subgroup
        # that keeps the fixed atoms, but for a spin-polarized cell or the
        # Fortran-derived routines) read the full grid
        # (periodic_sector_forces_refusal).  A band structure
        # keeps the sectors of the SCF under PARSEC_PERIODIC_SECTOR_BANDS=auto:
        # the bands read the potential the result holds on the full grid
        # (Output/post_scf.band_hand_over); under off the whole run reads the
        # full grid, as before.  nscf_kpoints runs no SCF here (Restart_Run is
        # required and the reference CLI runs it).
        moving_or_forces = (
            translation.relaxation is not None
            or translation.molecular_dynamics is not None
            or translation.compute_forces
        )
        sector_forces_refusal = (
            periodic_sector_forces_refusal(
                translation,
                supercell=_found_supercell is not None,
                # PARSEC_MOVING_SECTORS=auto: molecular dynamics and manual movements of a cell whose structures
                # are checked against the group of its sectors (SCF/moving_sectors.py), and relaxations with atoms
                # fixed by Move_Flag on the Fortran-derived routines in the sectors of their subgroup
                moving_sectors=moving_plan is not None,
                fortran_fixed_atoms=lift_fortran_fixed_atoms,
            )
            if moving_or_forces
            else None
        )
        if sector_forces_refusal is not None:
            # (what keeps the full grid keeps it for the moving runs of PARSEC_MOVING_SECTORS too)
            moving_plan = None
        periodic_sector_forces = moving_or_forces and sector_forces_refusal is None
        periodic_full_grid = None if periodic_sector_forces else (
            "a relaxation or molecular dynamics"
            if translation.relaxation is not None or translation.molecular_dynamics is not None
            else "forces"
            if translation.compute_forces
            else "a band structure"
            if translation.problem.bands is not None and periodic_sector_bands_setting() == "off"
            else "a non-self-consistent run"
            if getattr(translation, "nscf", None) is not None
            else None
        )
        wfndat_restart = translation.restart_run and getattr(translation, "restart_from_wfndat", False)
        # Under PARSEC_PERIODIC_SECTOR_FORCES=auto a refusal names what keeps
        # the full grid; under off the message of before.
        named_refusal = (
            moving_or_forces
            and periodic_full_grid is not None
            and sector_forces_refusal is not None
            and periodic_sector_forces_setting() == "auto"
        )
        if arguments.symmetry == "on" and (
            periodic_full_grid is not None or wfndat_restart or translation.problem.point_charges
        ):
            print(
                "Input error: --symmetry on cannot be used with "
                + (
                    (
                        "molecular dynamics"
                        if translation.molecular_dynamics is not None
                        else "a relaxation"
                        if translation.relaxation is not None
                        else "forces"
                    )
                    + f" of a periodic cell here, which keeps the full grid: {sector_forces_refusal}"
                    if named_refusal
                    else f"{periodic_full_grid} of a periodic cell, which the point-group sectors of a cell do not compute"
                    if periodic_full_grid is not None
                    else "Restart_From_Wfndat"
                    if wfndat_restart
                    else "point charges (Add_Point_Charges): the calculation uses no point group"
                ),
                file=sys.stderr,
            )
            return 2
        from .SCF.gamma_features import ts_ignore_symmetry_note

        # The Ignore_Symmetry that PARSEC demands of the TS correction (error 159) does not keep a bulk cell at the
        # Gamma point off the sectors of a CuPy run under PARSEC_GAMMA_TS_SECTORS=auto|on (SCF/gamma_features.py).
        ts_note = (
            ts_ignore_symmetry_note(
                translation, backend_is_cupy=lambda: _backend_resolves_to_cupy(arguments, translation)
            )
            if arguments.symmetry is None and periodic_full_grid is None and not wfndat_restart
            else None
        )
        if periodic_full_grid is not None:
            solver_symmetry_mode = "off"
        elif not wfndat_restart and not translation.problem.point_charges and (
            arguments.symmetry == "on" or (arguments.symmetry is None and not translation.ignore_symmetry)
        ):
            solver_symmetry_mode = arguments.symmetry or "auto"
        elif ts_note is not None and not translation.problem.point_charges:
            solver_symmetry_mode = "auto"
            restart_notes = [*restart_notes, ts_note]

    if moving_plan is not None:
        if not moving_plan.periodic:
            # (the NOTE of a cell names the group of its sectors: written after the preparation, by check_start)
            restart_notes = [*restart_notes, moving_plan.note()]
        if moving_plan.fixed is not None:
            # the atoms Move_Flag fixes in molecular dynamics, a species of their own for the detection
            fixed_atoms = moving_plan.fixed
    # A spin cell whose forces or moves the accelerated driver solves on the full grid (molecular dynamics, manual
    # movements, Move_Flag, Ignore_Symmetry): where both channels would not fit one device, one sector of the identity
    # whose basis the GPUs of the process share (PARSEC_SPIN_FALLBACK_SHARED; the routing kept the k-point driver for
    # a run that fits neither).
    spin_identity_sector = False
    if gamma_spin_on_device and solver_symmetry_mode == "off":
        from .SCF.gamma_spin import spin_full_grid_moves_plan

        moves_plan = spin_full_grid_moves_plan(translation, symmetry_off=_spin_symmetry_off(arguments, translation))
        if moves_plan is not None and moves_plan.fits:
            spin_identity_sector = moves_plan.layout == "shared"
            moves_note = moves_plan.moves_note()
            if moves_note is not None:
                restart_notes = [*restart_notes, f"NOTE: {moves_note}"]

    summary = summarize_translation(translation)
    if arguments.dry_run:
        try:
            pseudopotentials = load_pseudopotentials(
                translation.problem.pseudopotentials,
                xc_functional=translation.problem.scf.xc_functional,
            )
            domain_lines = domain_dry_run_lines(translation, pseudopotentials)
            selection = resolve_backend(arguments.backend, translation.problem)
            visible_gpus = _visible_gpus(resources) if selection.selected == "cupy" else None
        except (OSError, ValueError, BackendUnavailableError, NotImplementedError) as error:
            print(f"Validation error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for line in domain_lines:
            print(line)
        for warning in kept_group_warnings(translation.warnings, kept_reasons):
            print(f"WARNING: {warning}")
        print(
            f"Backend: requested={selection.requested}, "
            f"selected={selection.selected}"
        )
        for line in resources.dry_run_lines(selection.selected, visible_gpus):
            print(line)
        print(
            "Finite-difference builder: "
            f"{selection.finite_difference_builder}"
        )
        print(f"Hartree backend: {selection.hartree_backend}")
        print(
            f"Symmetry: mode={solver_symmetry_mode if kept_reasons or moving_plan is not None else symmetry_mode}; "
            "exact detection is deferred "
            "until grid construction"
        )
        if kept_reasons:
            print(kept_group_note(kept_reasons))
        if moving_plan is not None:
            print(moving_plan.note())
        elif moving_reason is not None:
            print(
                f"Molecular dynamics or manual movements (PARSEC_MOVING_SECTORS={moving_sectors_setting()}): "
                f"{moving_reason}"
            )
        for reason in selection.fallback_reasons:
            print(f"BACKEND FALLBACK: {reason}")
        print("Dry run successful; no grid or SCF calculation was started.")
        return 0

    if resources.device == "gpu":
        # A request for GPUs is held against what is there before any file
        # of the calculation is written: a number or a list against the GPUs
        # the process sees, and a plain request against the installation.
        # Whether a device answers is left to the preparation, which asks
        # the CUDA runtime beside its other work.
        try:
            if resources.names_gpus:
                _visible_gpus(resources)
            elif not _cupy_installed():
                raise BackendUnavailableError(
                    f"a GPU run was requested ({resources.device_source}) but CuPy is not "
                    "installed in this Python environment: install the CuPy build of your "
                    "CUDA version (python -m pip install cupy-cuda12x for CUDA 12), or run "
                    "with --device cpu"
                )
        except (ValueError, BackendUnavailableError) as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2

    input_directory = translation.source.parent
    # AFM_Scan: one folder per scan or part of a scan (rank 0 of an MPI run takes it).
    scan_folder_note = None
    if not arguments.dry_run and getattr(translation, "afm_scan", None) is not None:
        from parsec_python.Output.afm_scan import claim_scan_folder
        from .Parallel import mpi as _scan_mpi

        _scan_context = _scan_mpi.context()
        if _scan_context is None or _scan_context.is_root:
            try:
                scan_folder_note = claim_scan_folder(translation, input_directory)
            except (OSError, ValueError) as error:
                print(f"Input error: {error}", file=sys.stderr)
                return 2
    symmetry_cache_directory = _symmetry_cache_directory(arguments)
    # Output_File_Name names the report as in PARSEC; --log overrules it.
    log_path = _resolve_output(
        arguments.log, input_directory, translation.output_files.output_file_name
    )
    archive_path = _npz_path(
        _resolve_output(
            arguments.output,
            input_directory,
            "parsec_python_results.npz",
        )
    )
    protected_paths = {
        translation.source,
        *(item.path.resolve() for item in translation.problem.pseudopotentials.values()),
    }
    if log_path in protected_paths:
        print(f"Input error: refusing to overwrite input with log {log_path}", file=sys.stderr)
        return 2
    if not arguments.no_archive:
        if archive_path in protected_paths:
            print(
                f"Input error: refusing to overwrite input with archive {archive_path}",
                file=sys.stderr,
            )
            return 2
        if archive_path == log_path:
            print("Input error: --log and --output resolve to the same path", file=sys.stderr)
            return 2
    # parsec.dat, eigen.dat and the cube files of Output_Grid_Data, beside
    # the input as PARSEC writes them in its run directory.
    files = output_files_for(translation, input_directory)
    if _found_supercell is not None and _trim_route.cell_files:
        # PARSEC_KPOINT_SUPERCELL=auto: parsec.dat of the cell of the input, as the k-points write it (the
        # potential and density of the first cell, the k-points of the k-point route with the levels of each by their
        # Bloch phases, PARSEC_KPOINT_SUPERCELL_LEVELS; no orbitals: the supercell's states are no Bloch states of the
        # cell; Output_All_States and a Save_Wfn_Bands block keep the k-points, which write theirs),
        # which Restart_Run and nscf_kpoints of that input read on either route
        files.parsec_dat_view = _cell_parsec_dat_view(_mesh_problem, _found_supercell[1])
    taken = {*protected_paths, log_path}
    if not arguments.no_archive:
        taken.add(archive_path)
    conflict = output_file_conflict(files, taken)
    if conflict is not None:
        print(
            f"Input error: refusing to overwrite {conflict}: it is an input, the log "
            "or the archive of this run",
            file=sys.stderr,
        )
        return 2
    if (
        translation.restart_run
        and not getattr(translation, "restart_from_wfndat", False)
        and not (input_directory / "parsec.dat").is_file()
    ):
        print(
            f"Input error: Restart_Run: {input_directory / 'parsec.dat'} not found "
            "(PARSEC error 371)",
            file=sys.stderr,
        )
        return 2

    # The default rule of the sphere ran in the parser, before this clock:
    # its seconds belong to the calculation.  None for an input with a radius.
    started = time.perf_counter() - translation.domain_rule_seconds
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    reporter = None
    try:
        with run_log as log:
            reporter = AcceleratedTextReporter(
                log.write,
                translation,
                # A kept group (SCF/symmetry_off_inputs.py): the symmetry block of PARSEC 1.5 for this input, as the
                # reference CLI prints it, not that of Ignore_Symmetry (the sectors are named below it); a spin-orbit
                # run that allows symmetry: its own mode (PARSEC_SOC_SECTORS, _soc_symmetry_choice)
                symmetry_mode=None if kept_reasons else report_symmetry_mode,
            )
            if _found_supercell is not None and _trim_route.cell_files:
                # PARSEC_KPOINT_SUPERCELL=auto: the energy lines of the cell of the input, as the k-points give them
                reporter.reference.cells_in_report = int(np.prod(_found_supercell[1]))
            reporter.header()
            # What usrinputfile.F90 prints about the output keywords the input gives.
            for line in output_header_lines(translation.output_files):
                log.write(line)
            # usrinputfile.F90: the CAUTION line of Calc_Hartree_G (nscf_kpoints,
            # Create_DOS and Output_GW are routed to the reference CLI).
            from parsec_python.Output.post_processing_output import (
                post_processing_header_lines,
            )

            for line in post_processing_header_lines(translation):
                log.write(line)
            # PARSEC 1.5: FDET (Potential_Field) and pot.dat.
            from parsec_python.Output.correction_output import embedding_header_lines

            for line in embedding_header_lines(translation):
                log.write(line)
            if scan_folder_note is not None:
                log.write(scan_folder_note)
            for warning in kept_group_warnings(translation.warnings, kept_reasons):
                log.write(f"WARNING: {warning}")
            if symmetry_note is not None:
                log.write(f"WARNING: {symmetry_note}")
            for note in restart_notes:
                log.write(note)
            # Generate_Complete_Parsec_In (PARSEC 1.5): bak_complete_parsec.in beside the input (rank 0 of an MPI run)
            from parsec_python.Output.complete_input import write_complete_input
            from .Parallel import mpi as _mpi

            _context = _mpi.context()
            if _context is None or _context.is_root:
                for line in write_complete_input(translation):
                    log.write(line)
            log.write()
            if translation.relaxation is not None:
                log.write("\n".join(relaxation_header_lines(translation)))

            system = prepare_single_point(
                translation.problem,
                backend=arguments.backend,
                symmetry=solver_symmetry_mode,
                symmetry_cache_directory=symmetry_cache_directory,
                **({"spin_identity_sector": True} if spin_identity_sector else {}),
                symmetry_fixed_atoms=fixed_atoms,
                # forces and moves averaged over the group of the sectors: a small cell keeps its sectors
                # (PARSEC_PERIODIC_SECTOR_FLOOR applies to the SCF alone, and to the molecular dynamics and manual
                # movements of PARSEC_MOVING_SECTORS=auto, whose structures the full grid needs not average; on keeps
                # their sectors below it)
                sector_forces=periodic_sector_forces and (moving_plan is None or moving_sectors_setting() == "on"),
                # PARSEC_MOVING_SECTORS=auto: the pseudo-atoms of the state of the integrator or of the structures
                # of manual.dat, where they keep less than the atoms of a cluster, and the operations a cell's sectors
                # need at least
                **({"symmetry_constraints": moving_plan.constraints} if moving_plan is not None else {}),
                **(
                    {"minimum_sector_order": moving_plan.minimum_order}
                    if moving_plan is not None and moving_plan.minimum_order > 1
                    else {"minimum_sector_order": fortran_minimum_order}
                    if moving_plan is None and fortran_minimum_order > 1 and periodic_sector_forces
                    else {}
                ),
            )
            moving_tracker = None
            if moving_plan is not None:
                from .SCF.moving_sectors import MovingSectors

                def _prepare_moved(problem, *, symmetry, constraints, fixed):
                    return prepare_single_point(
                        problem,
                        backend=arguments.backend,
                        symmetry=symmetry,
                        symmetry_cache_directory=symmetry_cache_directory,
                        symmetry_fixed_atoms=fixed,
                        sector_forces=False,
                        symmetry_constraints=constraints,
                    )

                moving_tracker = MovingSectors(
                    moving_plan,
                    translation.problem,
                    prepare=_prepare_moved,
                    write=log.write,
                    time_step=(
                        None
                        if translation.molecular_dynamics is None
                        else translation.molecular_dynamics.time_step
                    ),
                )
                # the state of a restart and the structures of manual.dat against the group the driver found
                moving_tracker.check_start(system)
            if _context is None or _context.is_root:
                # What of a cluster with a point group runs on the full grid for want of the native extension
                # (also a NOTE line of parsec.out).
                for key, value in getattr(getattr(system, "backend_info", None), "details", ()):
                    if key == "symmetry_full_grid_note":
                        print(f"NOTE: {value}", file=sys.stderr)
            if accelerated_self_consistent_soc and soc_symmetry:
                # The orbitals of the spinor loop: the double-group sectors of PARSEC_SOC_SECTORS, which the
                # solver prepared without symmetry does not know of.
                system.backend_info = _soc_orbital_symmetry(system, soc_sectors, arguments.symmetry == "on")
            # Wavefunctions are much larger than all scalar observables and
            # are absent from the default archive.  Keep symmetry-sector
            # orbitals resident/lazy unless the user explicitly requests
            # their public full-grid array.  This does not affect SCF density,
            # energies, eigenvalues, or any normal text output.
            wavefunctions_wanted = bool(
                not arguments.no_archive
                and (
                    arguments.save_wavefunctions
                    or translation.output_all_states
                )
            )
            # A relaxation exports the orbitals of its last structure only.
            moving = (
                translation.relaxation is not None
                or translation.molecular_dynamics is not None
            )
            reporter.setup(system)
            if moving or translation.compute_forces:
                # A group that PARSEC_SYMMETRY_ATOM_TOLERANCE took beyond the
                # 1e-6 bohr of the forces and moves in the sectors: refused
                # here, before the SCF, not by the forces after it.
                from .Forces.sector_symmetry import sector_force_tolerance_refusal

                tolerance_refusal = sector_force_tolerance_refusal(system, moves=moving)
                if tolerance_refusal is not None:
                    raise ValueError(tolerance_refusal)
            # The forces of a cell solved in the sectors of a point group with
            # weighted representations: averaged over the full group G of the
            # sectors (Forces/sector_symmetry.py), which the ionic loop of a
            # relaxation finds once and keeps the moved positions in
            # (PARSEC_PERIODIC_SECTOR_FORCES=auto).
            sector_operations = None
            if periodic_sector_forces:
                from .Forces.sector_symmetry import sector_force_operations, sector_symmetry_operations

                sector_operations = sector_force_operations(system)
                # G of a point group, or H of Abelian sectors: the forces and
                # moved positions of a relaxation stay on the group the moved
                # atoms must keep (check_sector_atoms)
                if sector_symmetry_operations(system) is not None:
                    symmetrize_forces = True
                line = _periodic_sector_forces_line(system, sector_operations)
                if line is not None:
                    log.write(line)
                line = _fortran_sector_forces_line(
                    system, translation.movable if translation.relaxation is not None else None
                )
                if line is not None:
                    log.write(line)
                # A Gamma hybrid below the floor of its sectors on the full grid of the default routines
                # (PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto): its forces and moved positions averaged over the
                # operations of the cell, as the Fortran-derived routines average theirs there.
                line = _hybrid_full_grid_forces_line(system)
                if line is not None:
                    symmetrize_forces = True
                    log.write(line)
            # FDET (PARSEC 1.5): the frozen field and the tip sphere.
            from parsec_python.Output.correction_output import embedding_setup_lines

            for line in embedding_setup_lines(system):
                log.write(line)
            log.write("\n".join(_resource_report(resources, system)))
            if translation.problem.eigensolver.method == "none":
                # Eigensolver: none is PARSEC's TEST mode (parsec.F90:1085-1089):
                # the setup of the accelerated driver is reported and the run
                # ends without an SCF, as on the reference path.
                reporter.reference.test_mode_complete()
                log.write(
                    " Total accelerated Python wall time [sec] : "
                    f"{time.perf_counter() - started:11.2f}"
                )
                log.write(f"Text log: {log_path}")
                return 0
            restart_fields_read: list = []
            restart = load_restart_start(
                translation,
                system,
                log,
                input_directory,
                **({"fields_out": restart_fields_read} if restart_sectors is not None else {}),
            )
            if restart is not None and restart_sectors is not None:
                # A spin Restart_Run or Restart_From_Wfndat of a cluster in the sectors: kept where the stored
                # densities keep the group of the atoms, else prepared again as the switches off prepare it.
                from .SCF.symmetry_off_inputs import restart_in_sectors

                restart = restart_in_sectors(
                    translation,
                    system,
                    restart,
                    input_directory,
                    backend=arguments.backend,
                    symmetry=solver_symmetry_mode,
                    symmetry_cache_directory=symmetry_cache_directory,
                    log=log,
                    densities=restart_fields_read[0].densities if restart_fields_read else None,
                )
            del restart_fields_read
            if restart is not None:
                # wfn.dat: the stored orbitals go into the eigensolver of the run.
                from .SCF.restart import accelerated_restart_start

                restart = accelerated_restart_start(system, restart)
            # The wavefunctions of parsec.dat are brought to the host like
            # those of the archive.
            start_output_files(files, translation, system, log)
            if files.parsec_dat_view is not None:
                # the parsec.dat of the cell holds no orbitals: none brought to the host for it (those of the
                # supercell repeat no cell and are no Bloch states of the cell's k-points)
                from parsec_python.Output.output_files import WavefunctionSelection

                files.selection = WavefunctionSelection(mode="list", indices=(), explicit=False)
            wavefunctions_wanted = wavefunctions_wanted or files.selection_has_states
            system.materialize_final_wavefunctions = wavefunctions_wanted and not moving
            reporter = ReporterWithFiles(reporter, files)
            # Save_Intermediate_Charge_Density: parsec.dat after every SCF
            # iteration of every loop, with host fields.
            from .Output.iteration_fields import iteration_files_callback

            iteration_state_callback = iteration_files_callback(
                files, system, translation.problem
            )
            scf_keywords = (
                {}
                if iteration_state_callback is None
                else {"iteration_state_callback": iteration_state_callback}
            )
            if arguments.profile_operator:
                component_times = profile_hamiltonian_components(
                    system,
                    repeats=arguments.profile_repeats,
                )
                log.write(" Hamiltonian component microprofile [sec/apply]:")
                for name, value in component_times.items():
                    log.write(f" {name:30s} = {value:12.6f}")
                log.write()

            scf_started = time.perf_counter()
            if accelerated_self_consistent_soc and _asks_for_ionic_motion(translation):
                # PARSEC_SOC_DEVICE_FORCES=auto: the forces, relaxation or molecular dynamics of the spinor loop on the
                # device (its double-group sectors where the run allows symmetry, the forces where its spinors are),
                # reported and archived as the reference solver's spinor loops are.
                return _run_soc_ionic(
                    arguments,
                    translation,
                    system,
                    reporter=reporter,
                    files=files,
                    log=log,
                    soc_symmetry=soc_symmetry,
                    force_symmetry=soc_force_symmetry,
                    relax_restart=relax_restart,
                    dynamics_restart=dynamics_restart,
                    input_directory=input_directory,
                    archive_path=archive_path,
                    log_path=log_path,
                    started=started,
                    scf_started=scf_started,
                )
            if accelerated_self_consistent_soc:
                soc_projectors = tuple(
                    build_spin_orbit_projectors(
                        system.grid,
                        system.atoms,
                        system.pseudopotentials,
                        translation.problem.pseudopotentials,
                    )
                )
                result = run_self_consistent_soc(
                    system,
                    soc_projectors,
                    callback=lambda step: (
                        reporter.reference.iteration_self_consistent_soc(step),
                        files.iteration(step),
                    ),
                    note=log.write,
                    # --symmetry off, and Ignore_Symmetry without a mode on the command line, keep the whole grid
                    # with PARSEC_SOC_SECTORS too
                    symmetry=soc_symmetry,
                    # --symmetry on: PARSEC_SOC_SECTORS=auto takes the sector at every size
                    forced=arguments.symmetry == "on",
                )
                scf_completed = time.perf_counter()
                reporter.reference.finish_self_consistent_soc(
                    result, scf_completed - scf_started
                )
                files.scf_finished(result)
                reporting_completed = time.perf_counter()
                log.write(
                    f" Pre-SCF setup/reporting wall time [sec] : "
                    f"{scf_started - started:11.6f}"
                )
                log.write(
                    f" Post-SCF finalization/reporting [sec] : "
                    f"{reporting_completed - scf_completed:11.6f}"
                )
                log.write(
                    f" Total accelerated Python wall time [sec] : "
                    f"{time.perf_counter() - started:11.2f}"
                )
                if not arguments.no_archive:
                    saved = save_self_consistent_soc_result_archive(
                        archive_path,
                        result,
                        include_wavefunctions=(
                            arguments.save_wavefunctions
                            or translation.output_all_states
                        ),
                    )
                    log.write(f"Result archive: {saved}")
                files.finish(result, system.grid, translation.problem)
                log.write(f"Text log: {log_path}")
                return 0 if result.converged else 3


            def ionic_backend_of(extrapolation: str = "none"):
                backend = ionic_backend(extrapolation)
                if scf_keywords:
                    from functools import partial

                    backend = replace(
                        backend, run_scf=partial(backend.run_scf, **scf_keywords)
                    )
                if translation.problem.periodic_cell is not None:
                    from .Forces.periodic_device import with_periodic_operations

                    # A cell in the sectors of a point group takes the full
                    # group G of the sectors with its translations, what its
                    # forces are averaged over and its moved atoms keep.
                    backend = with_periodic_operations(backend)
                if restart is not None:
                    from parsec_python.SCF.restart import with_first_warm_start

                    backend = with_first_warm_start(backend, restart)
                return backend

            forces = None
            outcome = None
            dynamics = None
            polarization = None
            check_seconds = 0.0
            if getattr(translation, "polarizability_field", None) is not None:
                # The seven field loops of polar.f90 on the accelerated SCF,
                # reported and written to polar.dat as the reference CLI does.
                from parsec_python.cli import _run_polarizability

                from .SCF.polarizability import (
                    field_forces,
                    field_runner,
                    next_field_warm_start,
                )

                field_loop = None
                if "polarizability" in kept_reasons and solver_symmetry_mode != "off":
                    # PARSEC_POLARIZABILITY_SYMMETRY auto or on: the zero field in the group of the atoms, each field
                    # axis in the operations that keep it, the loops that are images of a solved one not solved.
                    from .SCF.symmetry_off_inputs import symmetric_field_loop

                    field_loop = symmetric_field_loop(
                        system,
                        problem=translation.problem,
                        backend=arguments.backend,
                        symmetry_cache_directory=symmetry_cache_directory,
                        log=log,
                        iteration_state_callback=iteration_state_callback,
                        symmetry=solver_symmetry_mode,
                    )
                polarization, result, forces = _run_polarizability(
                    system.reference,
                    translation,
                    reporter,
                    log,
                    input_directory,
                    scf_runner=field_runner(
                        system, iteration_state_callback=iteration_state_callback
                    ),
                    warm_start_builder=next_field_warm_start(system),
                    compute_forces=field_forces(system),
                    field_loop=field_loop,
                )
                scf_completed = time.perf_counter()
            elif translation.relaxation is not None:
                relaxation_backend = ionic_backend_of(translation.potential_extrapolation)
                if moving_tracker is not None:
                    # PARSEC_MOVING_SECTORS=auto: every structure of manual.dat checked against the group
                    relaxation_backend = moving_tracker.wrap(relaxation_backend)
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    relaxation_backend,
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=relax_restart,
                    symmetrize=symmetrize_forces,
                )
                result = outcome.result
                forces = outcome.records[-1].forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(outcome.system, result)
                scf_completed = time.perf_counter()
            elif translation.molecular_dynamics is not None:
                dynamics_backend = ionic_backend_of(translation.potential_extrapolation)
                if moving_tracker is not None:
                    # PARSEC_MOVING_SECTORS=auto: every step checked against the group, the state kept on it
                    dynamics_backend = moving_tracker.wrap(dynamics_backend)
                dynamics = run_recorded_molecular_dynamics(
                    translation,
                    system,
                    dynamics_backend,
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                    **({} if moving_tracker is None else {"after_step": moving_tracker.after_step}),
                )
                result, forces = dynamics.result, dynamics.forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(dynamics.system, result)
                scf_completed = time.perf_counter()
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, ionic_backend_of(), reporter, symmetrize=symmetrize_forces
                )
                scf_completed = time.perf_counter()
            else:
                result = run_scf(
                    system,
                    callback=reporter.iteration,
                    **({} if restart is None else {"warm_start": restart}),
                    **scf_keywords,
                )
                scf_completed = time.perf_counter()
                # run_scf makes the optional check of the Hartree boundary
                # before it returns: it is finalization, not SCF.
                check_seconds = hartree_boundary_check_seconds(result)
                reporter.finish(result, scf_completed - scf_started - check_seconds)
            if _found_supercell is not None and getattr(getattr(result, "energies", None), "total", None) is not None:
                # the mesh solved as its supercell: the energy of the cell of the input closes the report
                for line in _per_cell_lines(result.energies.total, _found_supercell[1], len(_mesh_problem.atoms),
                                            per_cell_report=_trim_route.cell_files):
                    log.write(line)
            reporting_completed = time.perf_counter()
            refinement = None
            if getattr(translation, "energy_refinement", None) is not None:
                # One non-self-consistent evaluation on the converged
                # orbitals, after everything the SCF reports; of the last
                # structure where the atoms moved.
                from parsec_python.Energy.refinement import refinement_report_lines
                from parsec_python.Output import RYDBERG_TO_EV

                from .refinement import refine_energy

                announce, progress = refinement_progress(
                    log.write, translation.energy_refinement
                )
                announce()
                refinement = refine_energy(
                    system if outcome is None else outcome.system,
                    result,
                    translation.energy_refinement,
                    progress=progress,
                )
                log.write(
                    "\n".join(refinement_report_lines(refinement, RYDBERG_TO_EV))
                )
            log.write(
                f" Pre-SCF setup/reporting wall time [sec] : "
                f"{scf_started - started:11.6f}"
            )
            log.write(
                f" Post-SCF finalization/reporting [sec] : "
                f"{reporting_completed - scf_completed + check_seconds:11.6f}"
            )
            if refinement is not None:
                log.write(
                    f" Post-SCF energy refinement [sec] : "
                    f"{time.perf_counter() - reporting_completed:11.6f}"
                )
            log.write(
                f" Total accelerated Python wall time [sec] : "
                f"{time.perf_counter() - started:11.2f}"
            )
            if not arguments.no_archive:
                saved = save_result_archive(
                    archive_path,
                    # PARSEC_KPOINT_SUPERCELL=auto: the archive of the cell of the input (its atoms, grid, fields
                    # and energies, the levels of the k-points by their Bloch phases), which --bands-from of that
                    # input reads
                    _cell_result_view(result, _mesh_problem, _found_supercell[1])
                    if _found_supercell is not None and _trim_route.cell_files
                    else result,
                    include_wavefunctions=(
                        arguments.save_wavefunctions
                        or translation.output_all_states
                    ),
                    # A plain single point keeps the call it always made.
                    **{
                        name: value
                        for name, value in (
                            ("forces", forces),
                            ("relaxation", outcome),
                            ("dynamics", dynamics),
                            ("refinement", refinement),
                            ("polarizability", polarization),
                        )
                        if value is not None
                    },
                )
                log.write(f"Result archive: {saved}")
            files.finish(result, system.grid, translation.problem)
            # Write_Potential_Field: pot.dat of PARSEC 1.5, of the last structure.
            from parsec_python.Output.pot_dat import write_potential_field_after_run

            write_potential_field_after_run(
                translation, system, result, outcome, dynamics, log.write, input_directory
            )
            # AFM_Scan: the table of the scan (afm_scan.dat).
            from parsec_python.Output.afm_scan import scan_lines_after_run

            scan_lines_after_run(translation, outcome, log.write, input_directory)
            # bandstruc of a Gamma-point cell, Create_DOS of a cluster: of the
            # last structure where the atoms moved (parsec.F90 calls calc_bands
            # after the movement loop, with the projectors of the last positions).
            last_system = (
                outcome.system
                if outcome is not None
                else dynamics.system
                if dynamics is not None
                else system
            )
            run_post_scf_outputs(last_system, translation, result, reporter, log.write, input_directory)
            log.write(f"Text log: {log_path}")
            if dynamics is not None:
                return 3 if dynamics.scf_failed else 0
            return relaxation_exit_code(
                result,
                outcome,
                manual=(
                    translation.relaxation is not None
                    and translation.relaxation.method == "manual"
                ),
            )
    except KeyboardInterrupt:
        if not getattr(run_log, "failure_reported", False) and not arguments.quiet:
            print("Calculation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        if arguments.debug:
            raise
        rank = getattr(arguments, "mpi_rank", None)
        if rank is not None:
            # an MPI run: the ranks other than 0 write no log, and a rank that fails alone ends the others
            # (the abort of main): every failed rank tells why, with its traceback, on the standard error
            import traceback

            text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            print(f"MPI rank {rank}: calculation failed:\n{text}", file=sys.stderr, end="", flush=True)
        elif not getattr(run_log, "failure_reported", False):
            print(
                f"Calculation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return 1
    finally:
        # The reporter may run a host thread beside the SCF: it ends here
        # where the calculation did not reach its report.
        if reporter is not None:
            reporter.close()


__all__ = ["main"]
