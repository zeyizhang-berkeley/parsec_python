"""Post-SCF outputs of the accelerated CLI: the band structure of a Gamma-point cell and the cluster DOS.

The accelerated driver runs the Gamma-point SCF of a periodic cell and the SCF of a cluster.  Two
outputs that PARSEC computes after such an SCF need nothing the accelerated result lacks, so they run
here, after the accelerated SCF, with the readable post-processing of the reference CLI:

``bandstruc`` (``calc_bands``) after a Gamma-point SCF of a scalar periodic cell
    :func:`parsec_python.cli._run_band_structure` with the converged potential of the accelerated
    result and the reference system it was prepared from (the accelerated periodic preparation *is*
    the reference preparation).  Its k-points go through the band machinery of
    :mod:`parsec_python.SCF.bands`, i.e. through the device k-point workers of
    :mod:`parsec_python.acceleration.SCF.kpoint_parallel` when a GPU is visible: the same eigensolves
    as the reference CLI makes after its own Gamma-point SCF, which runs on the host.  The SCF may
    solve in the sectors of the point group of the cell (``PARSEC_PERIODIC_SECTOR_BANDS=auto``):
    :func:`band_hand_over` gives the bands what a full-grid SCF gives them, and the k-points of the path
    are then solved in the sectors of their little groups under that group
    (``PARSEC_BAND_KPOINT_SECTORS=auto``, :mod:`..SCF.band_sectors`).

``Create_DOS`` of a cluster (``dos.F90`` l.149-168)
    :func:`parsec_python.cli._run_dos`: the histogram of the occupations of the accelerated levels.
    Nothing of it is worth a device (a histogram of ``States_Num`` levels).

Everything else these features combine with stays on the reference solver by name
(:func:`acceleration.cli._reference_only_features`): k-point SCFs (whose eigensolves already run on
the device workers), spin and spin-orbit bands, LDA+U, wires and slabs, Fortran's general grid, the
periodic DOS (it needs k-points), and the DOS under ``--mpi``.  Under ``--mpi`` every rank runs the
accelerated SCF of a Gamma cell (the SCF of one process), and the band k-points are dealt to the ranks
(``PARSEC_MPI_ACCELERATED_BANDS=auto``; ``off`` runs such an input on the reference solver, as before).
"""

from __future__ import annotations

import os
from pathlib import Path
import time

MPI_BANDS_SWITCH = "PARSEC_MPI_ACCELERATED_BANDS"


def _mpi_requested() -> bool:
    from ..Parallel import mpi

    return mpi.requested()


def mpi_accelerated_bands_setting() -> str:
    """``PARSEC_MPI_ACCELERATED_BANDS``: ``auto`` (the default since 2026-11-22) keeps the ``bandstruc`` of a scalar
    Gamma-point cell on the accelerated path under ``--mpi``: every rank runs the accelerated SCF that one process
    runs, and the band k-points are dealt to the ranks, each solved by the band solver of one process on the GPUs of its
    rank (:func:`..SCF.band_sectors.solve_band_sectors`; k-points solved whole by
    :func:`..SCF.kpoint_parallel.map_tasks`), so the levels are those of one process.  ``off``: such an input runs
    whole on the reference solver under ``--mpi``, as before: its SCF on the MPI k-point pools (the full grid, another
    route) and band levels tested on every state at ``Diag_Tolerance`` (216-atom diamond: levels 3.4e-5 Ry from one
    process, largest residual 9.8e-4 Ry, 669 s against 16 s)."""

    value = os.environ.get(MPI_BANDS_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{MPI_BANDS_SWITCH} must be auto or off, not {value!r}")
    return value


def accelerated_band_structure(translation) -> bool:
    """Whether the ``bandstruc`` of ``translation`` runs after the accelerated SCF.

    A scalar periodic cell sampled at Gamma only (the accelerated periodic SCF), not a wire or slab,
    without spin, spin-orbit species or LDA+U; under ``--mpi`` only with ``PARSEC_MPI_ACCELERATED_BANDS=auto``.
    Inputs that fail any of these keep the feature on the reference solver; most of them are routed there
    by another feature as well.
    """

    problem = translation.problem
    if problem.bands is None or problem.periodic_cell is None:
        return False
    scf = problem.scf
    if problem.has_kpoint_sampling or scf.spin_polarized:
        return False
    if scf.self_consistent_spin_orbit or scf.noncollinear:
        return False
    if getattr(problem.grid, "periodicity", 3) in (1, 2):
        return False
    for specification in problem.pseudopotentials.values():
        if specification.spin_orbit or getattr(specification, "has_hubbard", False):
            return False
    return not _mpi_requested() or mpi_accelerated_bands_setting() == "auto"


def accelerated_dos(translation) -> bool:
    """Whether ``Create_DOS`` of ``translation`` runs after the accelerated SCF: clusters, no ``--mpi``.

    The DOS of a periodic cell needs a k-point SCF (Monkhorst-Pack tetrahedra), which runs on the
    reference k-point drivers.
    """

    if getattr(translation, "dos", None) is None:
        return False
    return translation.problem.periodic_cell is None and not _mpi_requested()


class _FullGridLevels:
    """An SCF result with the level list and Fermi level of a full-grid SCF; every other attribute is the result's."""

    def __init__(self, result, eigenvalues, fermi_level) -> None:
        self._result = result
        self.eigenvalues = eigenvalues
        self.fermi_level = fermi_level

    def __getattr__(self, name: str):
        return getattr(self._result, name)


def band_hand_over(system, result):
    """The SCF result as ``calc_bands`` reads it: what a full-grid SCF of the cell would hand over.

    The bands read four things of the SCF (:func:`parsec_python.cli._run_band_structure`): the
    converged effective potential, the Fermi level, whether the loop converged and the number of
    states, which is the number of bands of every k-point of the path.  An SCF in the sectors of the
    point group of the cell holds the potential and the convergence as a full-grid SCF does: its
    result expands the fields of the orbits of the grid to every point (``to_public_scalar`` of the
    SCF loop).  Its level list repeats every weighted level for each partner, so the last one may
    complete a multiplet past ``States_Num``; the bands then get the ``States_Num`` lowest levels
    (the default count of the full grid without ``States_Num``), as many as the full grid solves,
    and the Fermi level of those levels as the SCF loop finds it on the full grid
    (:func:`parsec_python.Occupations.fermi_occupations` without weights): the levels past
    ``States_Num`` move the bracket of its bisection, which in a gap moves the level found by up
    to the gap under ``PARSEC_FERMI_GAP_ROOT=off``.  Its default places the level of a gap at the
    root of the count, which the levels past ``States_Num`` move by their thermal tails, and by
    ``(kT/2) ln(g_full/g_kept)`` where ``States_Num`` cuts partners of the lowest empty level (a
    six-fold level cut to 3 or 1: 0.35 or 0.90 kT): the level of the full grid's own list, which
    holds the same ``States_Num`` levels.  A result of the
    full grid, or of sectors whose list holds that count (whose Fermi
    level with weights is that of the full grid's list), is returned as it is.
    """

    import numpy as np

    from parsec_python.SCF.single_point import _number_of_states

    if getattr(result, "k_points", None) is not None:
        return result
    eigenvalues = getattr(result, "eigenvalues", None)
    if eigenvalues is None:
        return result
    reference = getattr(system, "reference", system)
    count = _number_of_states(reference)
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim != 1 or eigenvalues.size <= count:
        return result
    from parsec_python.Occupations import fermi_occupations

    levels = np.asarray(eigenvalues[:count], dtype=np.float64)
    fermi_level = fermi_occupations(
        levels, reference.electron_count, reference.input.scf.fermi_temperature_kelvin
    ).fermi_level
    return _FullGridLevels(result, levels, float(fermi_level))


def run_post_scf_outputs(system, translation, result, reporter, write, directory: Path) -> None:
    """Bands and DOS of the accelerated run, after the archive and the data files are written.

    ``system`` and ``result`` are the accelerated system and result of the SCF (of the last
    structure of an ionic loop), ``reporter`` the run's reporter (its ``reference`` is the
    :class:`parsec_python.Output.ParsecTextReporter` of the run), ``write`` the log.  Does nothing
    when the input asks for neither output, so the other runs are unchanged.
    """

    bands = accelerated_band_structure(translation)
    dos = accelerated_dos(translation)
    if not (bands or dos):
        return
    import parsec_python.cli as reference_cli
    from parsec_python.cli import _run_band_structure, _run_dos

    from ..Parallel import mpi

    started = time.perf_counter()
    text_reporter = getattr(reporter, "reference", reporter)
    reference_system = getattr(system, "reference", system)
    if bands:
        from ..SCF.band_sectors import band_kpoint_sectors_setting, band_symmetry_of, last_band_sector_record

        # The k-points of the path in the sectors of their little groups where the SCF ran in sectors
        # (PARSEC_BAND_KPOINT_SECTORS; off: every k-point on the whole grid, as before).
        symmetry = band_symmetry_of(system) if band_kpoint_sectors_setting() == "auto" else None
        # Under --mpi every rank solves its share of the band k-points and gets all the levels; rank 0 writes the
        # band files (bands.dat, bands_plot.dat, parsec_python_bands.npz), the others none.
        context = mpi.context()
        writes = reference_cli._WRITE_DATA_FILES
        if context is not None and not context.is_root:
            reference_cli._WRITE_DATA_FILES = False
        try:
            _run_band_structure(
                reference_system, translation, band_hand_over(system, result), text_reporter, directory,
                **({} if symmetry is None else {"symmetry": symmetry}),
            )
        finally:
            reference_cli._WRITE_DATA_FILES = writes
        record = last_band_sector_record() if symmetry is not None else {}
        if record:
            write(
                f" Band k-points in the sectors of their little groups: {record['split']} of {record['kpoints']} "
                f"k-points split, at most {record['largest']} sectors, under the {record['group']} operations of the "
                "SCF potential (PARSEC_BAND_KPOINT_SECTORS=auto)"
                + (
                    f"; {record['images']} sectors not solved, images of others (PARSEC_KPOINT_SECTOR_CLASSES=auto)"
                    if record.get("images")
                    else ""
                )
            )
    if dos:
        _run_dos(reference_system, translation, result, text_reporter, directory)
    write(f" Post-SCF band structure / DOS wall time [sec] : {time.perf_counter() - started:11.2f}")


__all__ = ["accelerated_band_structure", "accelerated_dos", "band_hand_over", "run_post_scf_outputs"]
