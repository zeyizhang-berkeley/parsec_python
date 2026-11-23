"""A k-point mesh of a cell and the Gamma point of its supercell: the same calculation, either way.

With the Bloch block of the finite-difference Laplacian as the kinetic term of a k-point
(:func:`parsec_python.Laplacian.finite_difference.build_bloch_negative_laplacian`, ``PARSEC_KPOINT_KINETIC=bloch``),
the Gamma-centred mesh of ``m_1 x m_2 x m_3`` points of a cell has the eigenvalues of the
Gamma point of the supercell of ``m_1 m_2 m_3`` cells on the same grid: the total energy of the supercell is ``M``
times that of the cell (``M = m_1 m_2 m_3``), to round-off.  Two routes use it, each behind its switch:

* ``PARSEC_KPOINT_SUPERCELL`` (:func:`trim_supercell_route`, :func:`supercell_of_trim_mesh`): a mesh of one or two
  points per axis (the time-reversal invariant k-points, ``Monkhorst_Pack_Shift`` zero or no shift block) is solved
  as the Gamma point of the supercell, whose orbitals are real and take the sectors of its translations by half the
  cell with the point group (``parsec_python.acceleration.Symmetry.axis_reflection``): real arithmetic and the
  little groups of the k-points.  ``auto`` (the default since 2026-11-19) takes it for a single-point SCF on the
  GPU where twice the estimated difference between the two kinetic forms (:func:`kinetic_form_difference_estimate`)
  is within ``PARSEC_KPOINT_SUPERCELL_TOLERANCE`` and the mesh reaches ``PARSEC_KPOINT_SUPERCELL_MINIMUM``; ``on``
  wherever the supercell can be solved (the ``auto`` of before); ``off`` keeps the k-points.
* ``PARSEC_SUPERCELL_KPOINTS=auto`` (:func:`kpoints_of_supercell`): a Gamma-point cell that repeats a smaller
  orthorhombic cell ``m_a`` times along each axis (a perfect crystal) is solved as that cell with the Gamma-centred
  mesh of ``m_1 x m_2 x m_3`` points: every translation of the crystal, through Bloch's theorem.  ``off`` by default.

The k-points of the default routines take the gradient form of the kinetic term (``-nabla^2 - 2ik.nabla + k^2``
with the gradient discretized on its own, PARSEC 1.5's form, ``PARSEC_KPOINT_KINETIC=gradient``), the supercell
the plain stencil of the grid, which is the Bloch form: the two discretizations of the same operator differ by
an amount that falls as a high power of the grid step (``h^{2M}`` for ``Expansion_Order`` ``2M``) and is zero at
the Gamma point.  :func:`kinetic_form_difference_estimate` evaluates it before the SCF from the pseudo-orbitals of
the atoms, and the default route takes the supercell only where it is small.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import os
import warnings

import numpy as np

TRIM_SWITCH = "PARSEC_KPOINT_SUPERCELL"
BLOCH_SWITCH = "PARSEC_SUPERCELL_KPOINTS"
TOLERANCE_SWITCH = "PARSEC_KPOINT_SUPERCELL_TOLERANCE"
MINIMUM_SWITCH = "PARSEC_KPOINT_SUPERCELL_MINIMUM"
GPUS_SWITCH = "PARSEC_KPOINT_SUPERCELL_GPUS"
STATES_SWITCH = "PARSEC_KPOINT_SUPERCELL_STATES"
#: The most GPUs of one process for which the default route takes the supercell (the command line; the full-SCF
#: runner divides the sectors of the supercell among its ranks and has no k-point route).  Measured on four A100
#: in one process (k-points against the supercell, peak GPU memory): 216 carbon atoms at a/18 1.37 against 4.27 GiB,
#: 216 silicon atoms at 66 points per edge 2.0 against 2.7 GiB, for 0 to 8 % in time; on one GPU the supercell held
#: less in every case measured.
DEFAULT_MOST_GPUS = 1
#: The default bound of the estimated kinetic-form difference per atom: 0.01 eV (in Ry).
DEFAULT_TOLERANCE = 0.01 / 13.605693122994
#: The estimate is compared at twice its value: the SCF differences measured on one A100 were 0.74 to 1.8 times
#: the estimate (silicon and diamond cubes at Expansion_Order 8 and 12, aluminium, MgO and an aluminium cell of
#: unequal steps at 12; 12 to 72 points per edge; 1.8 for diamond at 12 points, 1.36 to 1.46 from 14 on).
ESTIMATE_SAFETY = 2.0
_RYDBERG_EV = 13.605693122994


def _setting(name: str, values: tuple[str, ...], default: str) -> str:
    value = os.environ.get(name, default).strip().lower() or default
    if value not in values:
        raise ValueError(f"{name} must be {', '.join(values[:-1])} or {values[-1]}, not {value!r}")
    return value


def kpoint_supercell_setting() -> str:
    """``PARSEC_KPOINT_SUPERCELL``: ``auto`` (default), ``on`` or ``off``.

    ``auto``: a single-point SCF on the GPU of a mesh of time-reversal invariant k-points runs as the Gamma point of
    its supercell where that route applies, the estimated kinetic-form difference per atom, twice over, is within
    ``PARSEC_KPOINT_SUPERCELL_TOLERANCE`` and the mesh reaches ``PARSEC_KPOINT_SUPERCELL_MINIMUM``; ``on``: wherever
    the route applies, for any run (the ``auto`` of 2026-11-16 to 2026-11-18); ``off``: the k-points, as before
    2026-11-19.
    """
    return _setting(TRIM_SWITCH, ("auto", "on", "off"), "auto")


def kpoint_supercell_tolerance() -> float:
    """``PARSEC_KPOINT_SUPERCELL_TOLERANCE``: the bound (Ry per atom) of the default route; 0.01 eV by default."""
    raw = os.environ.get(TOLERANCE_SWITCH, "").strip()
    if not raw:
        return DEFAULT_TOLERANCE
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{TOLERANCE_SWITCH} must be a non-negative number (Ry per atom), not {raw!r}") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{TOLERANCE_SWITCH} must be a non-negative number (Ry per atom), not {raw!r}")
    return value


def kpoint_supercell_minimum() -> int:
    """``PARSEC_KPOINT_SUPERCELL_MINIMUM``: the floor of the default route in entries of the whole mesh (k-points
    times grid points times states of each), :data:`MINIMUM_MESH_ENTRIES` (``2^24``) by default; 0 takes every size."""
    raw = os.environ.get(MINIMUM_SWITCH, "").strip()
    if not raw:
        return MINIMUM_MESH_ENTRIES
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{MINIMUM_SWITCH} must be a non-negative integer (entries), not {raw!r}") from error
    if value < 0:
        raise ValueError(f"{MINIMUM_SWITCH} must be a non-negative integer (entries), not {raw!r}")
    return value


def kpoint_supercell_gpus() -> int:
    """``PARSEC_KPOINT_SUPERCELL_GPUS``: the most GPUs of one process for which ``auto`` takes the supercell,
    :data:`DEFAULT_MOST_GPUS` (one) by default."""
    raw = os.environ.get(GPUS_SWITCH, "").strip()
    if not raw:
        return DEFAULT_MOST_GPUS
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{GPUS_SWITCH} must be a positive integer (GPUs of one process), not {raw!r}") from error
    if value < 1:
        raise ValueError(f"{GPUS_SWITCH} must be a positive integer (GPUs of one process), not {raw!r}")
    return value


def kpoint_supercell_extra_states() -> int:
    """``PARSEC_KPOINT_SUPERCELL_STATES``: the levels per k-point that the supercell solves beyond the cell's states.

    ``cell`` (default) or ``0``: ``M`` times the cell's states, as before; the k-points of the cell's files then hold
    the fewest levels any k-point of the supercell's holds, which can be a few below ``States_Num`` (a
    ``Fix_Number_Eigenvalues`` restart from such a file stops with error 404).  A positive integer ``n``: ``M (N + n)``
    states, the levels the cell's files keep per k-point capped at ``N``, so that every k-point holds ``States_Num``
    where the ``n`` more per k-point reach its own; the ``M n`` more states are empty ones solved in every iteration.
    """
    raw = os.environ.get(STATES_SWITCH, "").strip().lower()
    if raw in ("", "cell"):
        return 0
    message = f"{STATES_SWITCH} must be cell or a non-negative integer (levels per k-point), not {raw!r}"
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(message) from error
    if value < 0:
        raise ValueError(message)
    return value


def process_gpu_count() -> int:
    """The GPUs this process computes on: those ``PARSEC_CUPY_DEVICES`` names (``auto``: every visible one;
    ``current``: one), as ``acceleration.Eigensolvers.symmetry.selected_device_ids`` reads it."""
    raw = os.environ.get("PARSEC_CUPY_DEVICES", "auto").strip().lower()
    if raw in {"current", "off"}:
        return 1
    if raw in {"", "auto"}:
        from parsec_python.acceleration.backends.cupy import cupy_device_count

        return int(cupy_device_count())
    return len({value.strip() for value in raw.split(",") if value.strip()})


def supercell_kpoints_setting() -> str:
    """``PARSEC_SUPERCELL_KPOINTS``: ``auto`` solves a repeating Gamma-point cell as its smallest cell with a mesh."""
    return _setting(BLOCH_SWITCH, ("auto", "off"), "off")


def validate_settings() -> None:
    """Read every switch of this module, so that a value it does not know is refused before the SCF."""
    from .supercell_levels import supercell_levels_setting

    kpoint_supercell_setting()
    kpoint_supercell_tolerance()
    kpoint_supercell_minimum()
    kpoint_supercell_gpus()
    kpoint_supercell_extra_states()
    supercell_kpoints_setting()
    # the levels of the cell's files (read when they are written, after the SCF)
    supercell_levels_setting()


def _bloch_kinetic_applies(problem) -> bool:
    """Whether the cell takes the Bloch kinetic term (a bulk orthorhombic cell of the default routines)."""
    grid = problem.grid
    if getattr(grid, "fortran_routines", False) or int(getattr(grid, "periodicity", 3)) != 3:
        return False
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    return bool(np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-12) and np.all(np.diag(lattice) > 0.0))


def _plain_scalar(problem) -> bool:
    return _scalar_refusal(problem) is None


def _scalar_refusal(problem) -> str | None:
    scf = problem.scf
    if scf.spin_polarized:
        return "spin polarization"
    if scf.noncollinear or scf.self_consistent_spin_orbit:
        return "spin-orbit coupling or non-collinear magnetism"
    if scf.is_hybrid:
        return "a hybrid functional"
    if scf.is_meta_gga:
        return "a meta-GGA functional"
    if getattr(scf, "fixed_occupations", None) is not None:
        return "fixed occupations (occup.in)"
    return None


def _grid_counts(problem) -> np.ndarray:
    from parsec_python.Grid.pbc import build_periodic_grid

    grid = build_periodic_grid(problem.periodic_cell, problem.grid)
    return np.asarray(grid.points_per_axis, dtype=np.int64)


def _same_grid(cell_problem, super_problem, repeats) -> bool:
    """Whether the supercell's grid is the cell's grid repeated (the same steps and shift)."""
    return bool(np.array_equal(_grid_counts(super_problem), _grid_counts(cell_problem) * np.asarray(repeats)))


def _trim_mesh_repeats(problem):
    """The repeats ``(m_1, m_2, m_3)`` of a Gamma-centred mesh of one or two points per axis, else ``None``."""
    if problem.periodic_cell is None:
        return None
    dims = getattr(problem, "monkhorst_pack_dimensions", None)
    shift = getattr(problem, "monkhorst_pack_shift", None)
    if dims is None or getattr(problem, "kpoints", None) is not None or shift is None:
        return None
    repeats = tuple(int(value) for value in dims)
    if any(value not in (1, 2) for value in repeats) or np.any(np.asarray(shift, dtype=np.float64) != 0.0):
        return None
    if repeats == (1, 1, 1):
        return None  # the Gamma point alone: nothing to fold
    return repeats


def _physics_refusal(problem) -> str | None:
    """Why the supercell cannot stand for the mesh (any setting), or ``None``."""
    grid = problem.grid
    if int(getattr(grid, "periodicity", 3)) != 3:
        return "a wire or slab"
    if getattr(grid, "fortran_routines", False):
        return "the Fortran-derived routines (Use_Fortran_Routines, or the input chose them)"
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    if not (np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-12) and np.all(np.diag(lattice) > 0.0)):
        return "an oblique cell (lattice vectors not along the Cartesian axes)"
    return _scalar_refusal(problem)


def _load_pseudopotentials(problem):
    """The pseudopotential files of ``problem`` by species label (the run reads and checks them itself)."""
    from parsec_python.V_ion.ionic_potential import load_pseudopotentials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_pseudopotentials(
            problem.pseudopotentials, xc_functional=problem.scf.xc_functional, check_correlation=False
        )


def _cell_states(problem, pseudopotentials=None) -> int:
    """The states of each k-point of the cell: ``States_Num``, else the default of the SCF (``ceil(N_e/2) + 6``,
    ``parsec_python.cli.planned_state_count``)."""
    requested = problem.scf.number_of_states
    if requested is not None:
        return int(requested)
    from parsec_python.V_ion.ionic_potential import ionic_charge

    tables = _load_pseudopotentials(problem) if pseudopotentials is None else pseudopotentials
    electrons = ionic_charge(tuple(problem.atoms), tables) - _cell_charge(problem)
    return int(np.ceil(0.5 * electrons)) + 6


def _cell_charge(problem) -> float:
    """``Net_Charges`` of the input: the charge of one cell (extensive: the supercell of ``M`` cells carries ``M``
    times as much, the cell of a repeating supercell ``1/M``)."""
    return float(getattr(problem.scf, "net_charge", 0.0) or 0.0)


#: The floor of the default route: the basis of the whole mesh (k-points times grid points times states of each)
#: below which the k-points are as fast or faster than the supercell, whose preparation (its group, sectors and
#: kernels) takes 5 to 10 s more.  It is the floor of the k-point sectors (``kpoint_sectors._MINIMUM_ENTRIES``).
#: Measured on one A100 with the 2 x 2 x 2 mesh (k-points against the supercell, s): 8-atom cubes of silicon at 40
#: points per edge (1.5e6 per k-point) 21.2/21.3, MgO at 40 20.6/25.3, diamond at 36 17.1/17.8, aluminium at 0.2 bohr
#: 18.4/22.5; silicon at 48 (2.7e6) 39.1/19.4, MgO at 56 (4.2e6) 55.3/27.3 and at 72 (9.0e6) 131.2/40.4; on four,
#: 64 silicon atoms at 44 (1.2e7) 24.7/21.9 and 216 carbon atoms at 54 (7.2e7) 32.2/29.8.
MINIMUM_MESH_ENTRIES = 1 << 24


def _supercell_problem(problem, repeats, states: int | None = None):
    """The supercell of ``problem`` repeated ``repeats`` times on the cell's grid steps, or ``None``.

    ``states``: those of each k-point of the cell (:func:`_cell_states`), ``M`` times as many in the supercell, with
    ``PARSEC_KPOINT_SUPERCELL_STATES`` more per k-point (:func:`kpoint_supercell_extra_states`).
    """
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    count = int(np.prod(repeats))
    if states is None:
        states = _cell_states(problem)
    atoms = [
        type(atom)(atom.symbol, tuple(np.asarray(atom.position, dtype=np.float64) + np.asarray(cell, float) @ lattice))
        for cell in np.ndindex(*repeats)
        for atom in problem.atoms
    ]
    counts = _grid_counts(problem)
    steps = np.diag(lattice) / counts
    base = replace(
        problem,
        atoms=atoms,
        periodic_cell=replace(problem.periodic_cell, lattice_vectors=lattice * np.asarray(repeats, float)[:, None]),
        # the sectors of the supercell's own group, not the density symmetrization of the k-point wedge with the
        # space group of the cell; Net_Charges is the charge of one cell, M cells carry M times as much (the
        # electrons of the supercell M times those of the cell, as the states)
        scf=replace(problem.scf, number_of_states=(int(states) + kpoint_supercell_extra_states()) * count,
                    symmetrize_density=False,
                    net_charge=_cell_charge(problem) * count),
        space_group=None,
        monkhorst_pack_dimensions=None,
        monkhorst_pack_shift=None,
    )
    # the cell's grid steps: the supercell's counts are the cell's times the repeats.  The shortest step first (the
    # spacing of before), then just below it, which the points rule of the input (the largest even count whose step
    # is not below the spacing) keeps on the same side of every step where the steps of the cell differ.
    shortest = float(np.min(steps))
    for spacing in (shortest, shortest * (1.0 - 1.0e-9)):
        supercell = replace(base, grid=replace(problem.grid, spacing=spacing))
        try:
            if _same_grid(problem, supercell, repeats):
                return supercell
        except ValueError:
            continue
    return None


def supercell_of_trim_mesh(problem, refusal=None):
    """``(supercell problem, repeats)`` for a Gamma-centred mesh of one or two points per axis, else ``None``.

    The supercell repeats the cell ``m_a`` times along each axis (the atoms, the lattice, ``M`` times the states),
    on the cell's grid steps, without k-points.  ``None`` under ``PARSEC_KPOINT_SUPERCELL=off``, for an explicit
    k-point list, a shifted mesh, a mesh of more points, an oblique cell,
    the Fortran-derived routines, a spin-polarized, spin-orbit, hybrid or meta-GGA calculation, fixed occupations,
    or a supercell for which ``refusal(supercell)`` (the accelerated runners' ``driver.periodic_sector_refusal``)
    gives a reason.  Neither the kind of run nor the kinetic-form estimate is looked at here
    (:func:`trim_supercell_route` adds them under ``auto``).
    """

    if kpoint_supercell_setting() == "off":
        return None
    repeats = _trim_mesh_repeats(problem)
    if repeats is None or _physics_refusal(problem) is not None:
        return None
    supercell = _supercell_problem(problem, repeats)
    if supercell is None:
        return None
    if refusal is not None and refusal(supercell) is not None:
        return None
    return supercell, repeats


# ----- the difference between the two kinetic forms ------------------------------------------------------------


def _symbols(order: int, step: float):
    """The finite-difference symbols of ``-d^2/dx^2`` and ``d/dx`` (times ``-i``) on a step ``step``."""
    from parsec_python.Laplacian.finite_difference import (
        first_derivative_coefficients,
        second_derivative_coefficients,
    )

    second = second_derivative_coefficients(order)
    first = first_derivative_coefficients(order)
    half = order // 2

    def laplacian(p):
        total = np.full(np.shape(p), second[half], dtype=np.float64)
        for j in range(1, half + 1):
            total += 2.0 * second[half + j] * np.cos(p * (j * step))
        return -total / step**2

    def derivative(p):
        total = np.zeros(np.shape(p), dtype=np.float64)
        for j in range(1, half + 1):
            total += 2.0 * first[half + j] * np.sin(p * (j * step))
        return total / step

    return laplacian, derivative


#: The grids of the estimate (1/bohr, bohr): beyond 40/bohr the pseudo-orbitals hold nothing that counts, and
#: the estimate changes by 2e-4 of itself from these steps to half of them (silicon, carbon, aluminium, MgO).
_P_STEP = 0.02
_P_MAX = 40.0
_R_STEP = 0.01


@lru_cache(maxsize=64)
def _axis_momentum_density(radii_key: bytes, wave_key: bytes, size: int, angular_momentum: int) -> np.ndarray:
    """``rho_1(p)`` on ``p = 0, dp, ...``: the momentum density of the ``m``-averaged orbital ``u_l(r)/r Y_lm``
    along one Cartesian axis, ``int rho_1 dp/(2 pi) = 1`` over the whole axis."""
    from scipy.interpolate import CubicSpline
    from scipy.special import spherical_jn

    radii = np.frombuffer(radii_key, dtype=np.float64, count=size)
    wave = np.frombuffer(wave_key, dtype=np.float64, count=size)
    order = np.argsort(radii, kind="stable")
    radii, wave = radii[order], wave[order]
    distinct = np.concatenate([[True], np.diff(radii) > 0.0])
    radii, wave = radii[distinct], wave[distinct]
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    kept = np.flatnonzero(np.abs(wave) > 1.0e-10 * peak)
    if peak == 0.0 or kept.size == 0:
        raise ValueError("an empty pseudo-wavefunction")
    # the orbital beyond 1e-10 of its peak holds nothing of the momentum density that counts
    r_end = float(radii[kept.max()])
    spline = CubicSpline(radii, wave)
    r = np.arange(_R_STEP, r_end + _R_STEP, _R_STEP)
    u = spline(r)
    p = np.arange(0.0, _P_MAX + 0.5 * _P_STEP, _P_STEP)
    # F_l(p) = int j_l(p r) u(r) r dr (the radial Hankel transform of R = u/r)
    transform = np.empty_like(p)
    weights = u * r * _R_STEP
    for start in range(0, p.size, 512):
        block = p[start:start + 512]
        transform[start:start + 512] = spherical_jn(angular_momentum, block[:, None] * r[None, :]) @ weights
    integrand = transform**2 * p**2
    norm = (2.0 / np.pi) * float(np.sum(0.5 * (integrand[1:] + integrand[:-1])) * _P_STEP)
    if not norm > 0.0:
        raise ValueError("a pseudo-wavefunction without norm")
    density = 4.0 * np.pi * transform**2 / norm  # int density d^3p/(2 pi)^3 = 1
    # rho_1(p) = (1/2 pi) int_{|p|}^inf density(q) q dq
    flux = density * p
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * _P_STEP)])
    return (cumulative[-1] - cumulative) / (2.0 * np.pi)


def _axis_term(rho: np.ndarray, order: int, step: float, k: float) -> float:
    """``<T_gradient - T_Bloch>`` along one axis at ``k`` for the symmetric momentum density ``rho`` (Ry)."""
    laplacian, derivative = _symbols(order, step)
    p = np.arange(rho.size, dtype=np.float64) * _P_STEP

    def difference(q):
        return laplacian(q - k) + 2.0 * k * derivative(q - k) + k * k - laplacian(q)

    values = rho * (difference(p) + difference(-p))  # the axis from -p_max to p_max, folded onto p >= 0
    return float(np.sum(0.5 * (values[1:] + values[:-1])) * _P_STEP / (2.0 * np.pi))


def _kpoint_kinetic_is_bloch(problem) -> bool:
    """Whether the k-points of this cell take the Bloch form themselves (``PARSEC_KPOINT_KINETIC``)."""
    from parsec_python.Laplacian.finite_difference import kpoint_kinetic_setting

    setting = kpoint_kinetic_setting()
    if setting == "gradient":
        return False
    from parsec_python.Grid.pbc import build_periodic_grid
    from parsec_python.Laplacian.finite_difference import _wrapped_box

    return bool(_wrapped_box(build_periodic_grid(problem.periodic_cell, problem.grid)))


def kinetic_form_difference_estimate(problem, pseudopotentials=None) -> float | None:
    """The estimated ``E(k-points) - E(supercell)`` per atom (Ry) of a mesh of time-reversal invariant points.

    First order in the difference of the two kinetic operators, ``sum_nk f_nk <u_nk|T_gradient - T_Bloch|u_nk>``,
    with the momentum density of the occupied states taken from the pseudo-wavefunctions of the atoms (their
    occupations, averaged over ``m``): along an axis of step ``h`` where half the mesh sits at ``k = pi/L``, a plane
    wave ``e^{ipx}`` of the orbital has the kinetic energy ``s(p)`` in the supercell (the symbol of the stencil of
    ``Expansion_Order``) and ``s(p - k) + 2k d(p - k) + k^2`` at the k-point (``d``: the symbol of the
    first-derivative stencil).  Zero where the k-points take the Bloch form themselves
    (``PARSEC_KPOINT_KINETIC=bloch`` or ``auto`` on a cell of one grid step).  ``None`` where a species has no
    pseudo-wavefunction of an occupied channel.  ``pseudopotentials``: the loaded files by species label (read from
    the problem where not given).
    """

    repeats = _trim_mesh_repeats(problem)
    if repeats is None:
        return 0.0
    if _kpoint_kinetic_is_bloch(problem):
        return 0.0
    if pseudopotentials is None:
        pseudopotentials = _load_pseudopotentials(problem)
    lattice = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    steps = lattice / _grid_counts(problem)
    order = int(problem.grid.expansion_order)
    per_species: dict[str, float] = {}
    for label, potential in pseudopotentials.items():
        total = 0.0
        for angular_momentum, occupation in sorted(potential.channel_occupations.items()):
            if not occupation:
                continue
            wave = potential.radial_wavefunctions.get(angular_momentum)
            if wave is None:
                return None
            radii = np.ascontiguousarray(potential.radii, dtype=np.float64)
            wave = np.ascontiguousarray(wave, dtype=np.float64)
            try:
                rho = _axis_momentum_density(radii.tobytes(), wave.tobytes(), int(radii.size), int(angular_momentum))
            except ValueError:
                return None
            for axis in range(3):
                if repeats[axis] == 2:
                    # half the points of the mesh sit at k = pi/L along this axis
                    total += 0.5 * float(occupation) * _axis_term(rho, order, float(steps[axis]),
                                                                 math.pi / float(lattice[axis]))
        per_species[label] = total
    atoms = list(problem.atoms)
    if not atoms:
        return 0.0
    if any(atom.symbol not in per_species for atom in atoms):
        return None
    return float(sum(per_species[atom.symbol] for atom in atoms) / len(atoms))


# ----- the route ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimRoute:
    """What :func:`trim_supercell_route` decided: the supercell (``problem``, ``repeats``) or the k-points.

    ``estimate`` is the estimated kinetic-form difference per atom (Ry) where it was evaluated, ``note`` the line
    the output prints (``None``: nothing to say, the input is not a mesh of time-reversal invariant points or the
    switch is ``off``).
    """

    problem: object | None
    repeats: tuple[int, int, int] | None
    estimate: float | None
    note: str | None
    # the files and the report of the cell of the input (``auto``: ``parsec.dat``, the result archive and the
    # energies of the report per cell), or those of the supercell (``on``, as before)
    cell_files: bool = False

    @property
    def taken(self) -> bool:
        return self.problem is not None


def _run_refusal(translation) -> str | None:
    """Why the default route keeps the k-points for this kind of run: what reads or writes the atoms, the
    k-points or the grid of the cell, and what the supercell's output would report differently."""
    problem = translation.problem
    checks = (
        (getattr(translation, "relaxation", None) is not None, "a relaxation"),
        (getattr(translation, "molecular_dynamics", None) is not None, "molecular dynamics"),
        (bool(getattr(translation, "compute_forces", False)), "forces (Skip_Force: .false.)"),
        (getattr(problem, "bands", None) is not None, "a band structure"),
        (getattr(translation, "nscf", None) is not None, "a non-self-consistent run"),
        (getattr(translation, "dos", None) is not None, "Create_DOS"),
        (getattr(translation, "gw", None) is not None, "the BerkeleyGW output"),
        (bool(getattr(translation, "restart_run", False)), "Restart_Run"),
        (getattr(translation, "polarizability_field", None) is not None, "Polarizability"),
        (getattr(translation, "energy_refinement", None) is not None, "Energy_Refinement"),
        (getattr(translation, "mlff", None) is not None, "Initial_Structure: mlff"),
        (bool(getattr(translation, "fix_number_eigenvalues", False)), "Fix_Number_Eigenvalues"),
        (bool(getattr(translation, "write_potential_field", False)), "Write_Potential_Field"),
        (getattr(translation, "afm_scan", None) is not None, "AFM_Scan"),
        (bool(getattr(getattr(translation, "output_files", None), "grid_data", ())), "Output_Grid_Data"),
        (bool(getattr(getattr(translation, "output_files", None), "intermediate_density", False)),
         "Save_Intermediate_Charge_Density"),
        (bool(getattr(problem, "point_charges", None)), "point charges"),
        (float(getattr(translation, "magnetic_field_gauss", 0.0) or 0.0) != 0.0, "a magnetic field"),
        (bool(getattr(translation, "complex_wavefunctions", False)), "complex wavefunctions"),
        # the orbitals of the archive: those of the supercell are not the k-points' Bloch states
        (bool(getattr(translation, "output_all_states", False)), "Output_All_States"),
    )
    for found, reason in checks:
        if found:
            return reason
    return None


def _format(value: float) -> str:
    return f"{value:.2e} Ry ({value * _RYDBERG_EV:.2e} eV)"


def trim_supercell_route(
    translation,
    refusal=None,
    *,
    pseudopotentials=None,
    device: str | None = None,
    kpoint_ranks: bool = False,
    gpus=None,
    orbitals: bool = False,
) -> TrimRoute:
    """The route of a mesh of time-reversal invariant k-points: the supercell at its Gamma point, or the k-points.

    ``translation`` is the parsed input (its ``problem`` and the kind of run), ``refusal`` the callback of the
    accelerated runners (``driver.periodic_sector_refusal``) that names what the Gamma route of the supercell does
    not solve.  Under ``PARSEC_KPOINT_SUPERCELL=auto`` (the default) the supercell is taken for a single-point SCF
    (no forces, movements, bands, restart and the other runs :func:`_run_refusal` names) whose estimated difference
    of the two kinetic forms (:func:`kinetic_form_difference_estimate`), times :data:`ESTIMATE_SAFETY`, is within
    ``PARSEC_KPOINT_SUPERCELL_TOLERANCE`` and whose whole mesh (k-points times grid points times states) reaches
    ``PARSEC_KPOINT_SUPERCELL_MINIMUM``; under ``on`` wherever the supercell can be solved; under ``off`` never.
    ``device`` is the request of the command line (``--device``): under ``auto`` a run on the CPU (``cpu``, or no
    usable CuPy) keeps the k-points, whose reference solver is the faster one there; so does ``kpoint_ranks``, a
    run of the command line with ``--mpi``, whose ranks divide the k-points (each would solve the whole supercell;
    the full-SCF runner divides its sectors).  ``gpus`` is the number of GPUs of this process (or a function that
    counts them, called only where the count decides, :func:`process_gpu_count`): under ``auto`` more than
    ``PARSEC_KPOINT_SUPERCELL_GPUS`` (one by default) keep the k-points, whose pools divide the k-points over the
    GPUs while the largest sectors of the supercell share one (``None``: not looked at, the full-SCF runner, which
    has no k-point route).  ``orbitals``: the orbitals are to be saved (``--save-wavefunctions``), which those of
    the supercell are not (the k-points under ``auto``).  The decision depends on the input and the request alone,
    so every MPI rank takes the same one.
    """

    setting = kpoint_supercell_setting()
    tolerance = kpoint_supercell_tolerance()
    minimum = kpoint_supercell_minimum()
    most_gpus = kpoint_supercell_gpus()
    problem = translation.problem
    if setting == "off":
        return TrimRoute(None, None, None, None)
    repeats = _trim_mesh_repeats(problem)
    if repeats is None:
        return TrimRoute(None, None, None, None)
    mesh = " x ".join(str(value) for value in repeats)

    def keep(reason: str, estimate=None) -> TrimRoute:
        return TrimRoute(None, None, estimate, (
            f"NOTE: the k-point mesh {mesh} of time-reversal invariant points runs as k-points, not as the Gamma "
            f"point of its supercell (PARSEC_KPOINT_SUPERCELL={setting}): {reason}"
        ))

    reason = _physics_refusal(problem)
    if reason is not None:
        return keep(reason)
    if setting == "auto":
        reason = _run_refusal(translation)
        if reason is None and orbitals:
            reason = "the orbitals saved (--save-wavefunctions)"
        if reason is not None:
            return keep(reason + " (the default route takes single-point energies only; "
                        "PARSEC_KPOINT_SUPERCELL=on takes the supercell for it)")
        if device is not None and not _uses_gpu(device):
            return keep("a run on the CPU, where the reference k-point solver is the faster "
                        "(PARSEC_KPOINT_SUPERCELL=on takes the supercell)")
        if kpoint_ranks:
            return keep("--mpi, whose ranks divide the k-points (benchmarks/mpi_full_scf.py divides the sectors of "
                        "the supercell; PARSEC_KPOINT_SUPERCELL=on takes it here)")
        if gpus is not None:
            try:
                count = int(gpus() if callable(gpus) else gpus)
            except Exception as error:  # noqa: BLE001 - no CUDA to count: the run reports it itself
                return keep(f"the GPUs of this process could not be counted ({error})")
            if count > most_gpus:
                return keep(
                    f"{count} GPUs in one process, more than {GPUS_SWITCH} = {most_gpus}: the pools of the "
                    "k-points divide them over the GPUs, while the largest sectors of the supercell share one, "
                    "which then holds more (PARSEC_KPOINT_SUPERCELL=on takes the supercell)"
                )
    try:
        if pseudopotentials is None:
            pseudopotentials = _load_pseudopotentials(problem)
        states = _cell_states(problem, pseudopotentials)
    except (OSError, ValueError):
        # a file the run itself reads and refuses later, by name
        return keep("its pseudopotential files could not be read here")
    if setting == "auto":
        entries = int(np.prod(repeats)) * int(np.prod(_grid_counts(problem))) * int(states)
        if entries < minimum:
            return keep(
                f"a small cell (k-points x grid points x states = {entries:.3g} < {MINIMUM_SWITCH} = {minimum:.3g}, "
                "where the k-points are as fast or faster; PARSEC_KPOINT_SUPERCELL=on takes the supercell)"
            )
    supercell = _supercell_problem(problem, repeats, states)
    if supercell is None:
        return keep("the supercell's grid would not repeat the cell's (steps that differ too much between the axes)")
    if refusal is not None:
        reason = refusal(supercell)
        if reason is not None:
            return keep(f"the Gamma point of the supercell does not solve {reason}")
    try:
        estimate = kinetic_form_difference_estimate(problem, pseudopotentials)
    except (OSError, ValueError):
        # a file the run itself reads and refuses later, by name
        estimate = None
    if setting == "auto":
        if estimate is None:
            return keep("no estimate of the difference of the kinetic forms (a species without pseudo-wavefunctions)")
        if ESTIMATE_SAFETY * abs(estimate) > tolerance:
            return keep(
                f"the estimated difference of the two kinetic forms, {_format(estimate)} per atom, is above "
                f"{TOLERANCE_SWITCH}/{ESTIMATE_SAFETY:g} = {_format(tolerance / ESTIMATE_SAFETY)} (a finer grid or a "
                "higher Expansion_Order lowers it; PARSEC_KPOINT_SUPERCELL=on takes the supercell)"
            )
    return TrimRoute(supercell, repeats, estimate,
                     describe("supercell", repeats, estimate=estimate, per_cell=setting == "auto"),
                     cell_files=setting == "auto")


def _uses_gpu(device: str) -> bool:
    """Whether a run of the request ``device`` (``auto``, ``gpu`` or ``cpu``) runs on the GPU."""
    request = str(device).strip().lower()
    if request == "cpu":
        return False
    if request == "gpu":
        return True
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # noqa: BLE001 - no accelerated package or no CUDA: the CPU
        return False


def kpoints_of_supercell(problem, tolerance: float = 1.0e-7):
    """``(cell problem, repeats)`` for a Gamma-point cell that repeats a smaller one, else ``None``.

    The repeats ``m_a`` are the largest whole divisions of each axis whose translation maps the atoms (species by
    species, modulo the cell); the smaller cell keeps the atoms in it, ``1/M`` of ``Net_Charges`` and of the states
    (at least as many per k-point, rounded up, and six more for the filters), the grid steps of the supercell, and the
    Gamma-centred mesh ``m_1 x m_2 x m_3``.  ``None`` under ``PARSEC_SUPERCELL_KPOINTS=off``, with k-points already,
    without a repeat, where the cell's grid would not repeat into the supercell's or the builder refuses it (a cell
    too small for the stencil), under an explicit ``PARSEC_KPOINT_KINETIC=gradient`` (the caller runs the mesh with
    ``bloch``, which reproduces the supercell), and for what :func:`supercell_of_trim_mesh` refuses besides the mesh.
    The implementation is :func:`parsec_python.SCF.supercell_mesh.smallest_cell` (since 2026-11-19, with the meshes
    of ``PARSEC_SUPERCELL_KPOINTS_MESH``).
    """

    from .supercell_mesh import smallest_cell

    return smallest_cell(problem, tolerance)


def _repeat_of_axis(positions_by_species, side: float, axis: int, points: int, tolerance: float) -> int:
    """The largest ``m`` dividing the grid's ``points`` along ``axis`` (and every species' atom count) whose
    translation ``side/m`` maps the atoms onto themselves, modulo the cell (1 where none does)."""
    from scipy.spatial import cKDTree

    def _has_tolerance_perfect_matching(moved, points_, tolerance_, period_):
        # a translation of a crystal maps every atom onto one atom of its species, a different one each
        tree = cKDTree(np.mod(points_, period_), boxsize=period_)
        distances, nearest = tree.query(np.mod(moved, period_), k=1, distance_upper_bound=2.0 * tolerance_)
        return bool(np.all(np.isfinite(distances)) and np.unique(nearest).size == len(points_))

    period = np.asarray(positions_by_species["__period__"])
    species = {name: points_ for name, points_ in positions_by_species.items() if name != "__period__"}
    for m in sorted((value for value in range(2, points + 1) if points % value == 0), reverse=True):
        if any(len(points_) % m for points_ in species.values()):
            continue
        shift = np.zeros(3)
        shift[axis] = side / m
        if all(_has_tolerance_perfect_matching(points_ + shift, points_, tolerance, period) for points_ in species.values()):
            return m
    return 1


# ----- the output of the supercell ---------------------------------------------------------------------------


def describe(route: str, repeats, *, estimate: float | None = None, per_cell: bool = False) -> str:
    """The note of the output: what was solved and the factor between the two total energies.

    ``per_cell``: the report gives the energies of the cell of the input (``auto``), else those of the supercell
    (``on``)."""
    m = tuple(int(value) for value in repeats)
    count = int(np.prod(m))
    if route == "supercell" and per_cell:
        from .supercell_levels import supercell_levels_setting

        text = (
            f"NOTE: the Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} of the cell is solved as the Gamma point "
            f"of its supercell of {count} cells (PARSEC_KPOINT_SUPERCELL={kpoint_supercell_setting()}; the Bloch "
            f"kinetic term): the energies reported are those of the input cell (1/{count} of the supercell's, whose "
            "total closes the output), the levels those of the supercell (the union of the mesh's); parsec.dat and "
            "the result archive are the cell's"
            # PARSEC_KPOINT_SUPERCELL_LEVELS=off keeps the sentence of before (its levels dealt in turn)
            + (", the levels of each k-point by their Bloch phases" if supercell_levels_setting() == "auto" else "")
        )
        if estimate is not None:
            text += (
                f"; the k-points with the gradient kinetic term (PARSEC_KPOINT_SUPERCELL=off) would give "
                f"{_format(estimate)} per atom more, by the estimate before the SCF"
            )
        return text
    if route == "supercell":
        text = (
            f"NOTE: the Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} of the cell is solved as the Gamma point "
            f"of its supercell of {count} cells (PARSEC_KPOINT_SUPERCELL={kpoint_supercell_setting()}; the Bloch "
            f"kinetic term): the total energy and the levels reported are those of the supercell, {count} times the "
            "energy per cell of the mesh (the energy per cell closes the output)"
        )
        if estimate is not None:
            text += (
                f"; the k-points with the gradient kinetic term (PARSEC_KPOINT_SUPERCELL=off) would give "
                f"{_format(estimate)} per atom more, by the estimate before the SCF"
            )
        return text
    return (
        f"the cell repeats a smaller one {m[0]} x {m[1]} x {m[2]} times: it is solved as that cell with the "
        f"Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} (PARSEC_SUPERCELL_KPOINTS=auto; the Bloch kinetic "
        f"term): the total energy reported is that of the smaller cell, 1/{count} of the supercell's"
    )


def per_cell_lines(total_energy: float, repeats, cell_atoms: int, *, per_cell_report: bool = False) -> list[str]:
    """The closing lines of a mesh solved as its supercell: the total energy per cell and per atom of the input.

    ``total_energy`` is the supercell's.  ``per_cell_report``: the energy lines of the report above were those of
    the cell (``auto``, :func:`energies_per_cell`), and the supercell's total is given here."""
    count = int(np.prod(tuple(int(value) for value in repeats)))
    per_cell = float(total_energy) / count
    lines = [
        f" Total energy of the input cell (the k-point mesh, 1/{count} of the supercell) = {per_cell:20.8f} [Ry]",
    ]
    if per_cell_report:
        lines.append(
            f" Total energy of the supercell of {count} cells (its Gamma point solved; the energies above are those"
            f" of the input cell) = {float(total_energy):20.8f} [Ry]"
        )
    if cell_atoms:
        lines.append(
            f" Total energy per atom of the input cell = {per_cell / cell_atoms:20.8f} [Ry]"
            f" = {per_cell / cell_atoms * _RYDBERG_EV:16.8f} [eV]"
        )
    return lines


def mesh_kpoints(problem, repeats) -> np.ndarray:
    """The Cartesian k-points (1/bohr) of the Gamma-centred mesh ``repeats`` of the cell of ``problem``, in the order
    of :func:`numpy.ndindex`."""
    sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    return np.array(
        [[index[axis] / repeats[axis] * 2.0 * math.pi / sides[axis] for axis in range(3)] for index in np.ndindex(*repeats)],
        dtype=np.float64,
    ).reshape(-1, 3)


def cell_parsec_dat_view(cell_problem, repeats):
    """The transform of ``parsec.dat`` of a mesh solved as its supercell: ``(result, grid, problem)`` of the
    supercell to those of the cell of the input, as the k-points would write them.

    The potential and the density are those of the first cell of the supercell (they repeat with the cell); the
    k-points, their weights and levels those of the k-point route (:func:`parsec_python.SCF.supercell_levels.
    cell_levels`: every state of the supercell at the k-point of its Bloch phases, the irreducible wedge of the
    cell; under ``PARSEC_KPOINT_SUPERCELL_LEVELS=off``, or where the orbitals do not tell, the whole mesh with the
    levels dealt to the k-points in turn, whose union is the mesh's); the file holds no orbitals (the supercell's
    states are no Bloch states of the cell; the k-point route writes those of every k-point,
    ``PARSEC_KPOINT_WAVEFUNCTIONS``).  ``Restart_Run`` of the cell reads it on either route.
    """
    from types import SimpleNamespace

    from parsec_python.Grid.pbc import build_periodic_grid

    repeats = tuple(int(value) for value in repeats)
    cell_grid = build_periodic_grid(cell_problem.periodic_cell, cell_problem.grid)
    counts = np.asarray(cell_grid.points_per_axis, dtype=np.int64)

    def view(result, grid, problem):
        from .supercell_levels import cell_levels

        for name in ("eigenvalues", "occupations", "next_effective_potential", "density"):
            if getattr(result, name, None) is None:
                raise ValueError(f"this result keeps no {name}, which parsec.dat needs")
        # every level at the k-point of its Bloch phases (the blocks of the k-points one after another)
        levels = cell_levels(result, cell_problem, repeats)
        stand_in = SimpleNamespace(
            eigenvalues=levels.eigenvalues,
            occupations=levels.occupations,
            next_effective_potential=cell_block(result.next_effective_potential, counts, repeats),
            density=cell_block(result.density, counts, repeats),
            k_points=levels.k_points,
            k_weights=levels.weights,
            wavefunctions=None,
            parsec_dat_notes=() if levels.note is None else (levels.note,),
        )
        return stand_in, cell_grid, cell_problem

    return view


#: Fields of an energy breakdown that are no energies (not divided by the number of cells).
_NON_ENERGY_FIELDS = frozenset({"exact_exchange_fraction", "linearized"})


def energies_per_cell(energies, count: int):
    """The energy breakdown of the supercell of ``count`` cells as that of one cell: every energy over ``count``."""
    from dataclasses import fields, is_dataclass

    count = int(count)
    if count == 1 or energies is None or not is_dataclass(energies):
        return energies
    changes = {}
    for item in fields(energies):
        value = getattr(energies, item.name)
        if item.name in _NON_ENERGY_FIELDS or isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (float, np.floating)):
            changes[item.name] = float(value) / count
    return replace(energies, **changes)


class _CellResult:
    """A result of the supercell seen as that of the cell of the input: the attributes of ``overrides``, the rest
    of ``result``."""

    def __init__(self, result, overrides: dict) -> None:
        self.__dict__["_result"] = result
        self.__dict__["_overrides"] = overrides

    def __getattr__(self, name: str):
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(self.__dict__["_result"], name)


def _dealt_levels(eigenvalues, count: int) -> np.ndarray:
    """The order that deals the levels of the supercell to the ``count`` k-points of the mesh in turn from the lowest
    (the union is the mesh's, the assignment to each k-point not): :func:`.supercell_levels.dealt_levels`."""
    from .supercell_levels import dealt_levels

    return dealt_levels(eigenvalues, count)


def cell_result_view(result, cell_problem, repeats):
    """The result of a mesh solved as its supercell as the k-points of the cell of ``cell_problem`` would give it,
    for the result archive (``parsec_python_results.npz``) under ``auto``: the atoms and the grid of the cell, the
    fields of its first cell, the k-points, weights and levels of :func:`cell_parsec_dat_view`, the energies and the
    electrons of one cell, no orbitals.  ``--bands-from`` of that input reads it."""
    from parsec_python.Grid.pbc import build_periodic_grid

    repeats = tuple(int(value) for value in repeats)
    count = int(np.prod(repeats))
    cell_grid = build_periodic_grid(cell_problem.periodic_cell, cell_problem.grid)
    counts = np.asarray(cell_grid.points_per_axis, dtype=np.int64)
    supersize = int(np.prod(counts)) * count

    def host(values):
        return values.get() if hasattr(values, "get") and not isinstance(values, np.ndarray) else values

    def block(values):
        if values is None:
            return None
        array = np.asarray(host(values))
        if array.ndim != 1 or array.shape[0] != supersize:
            return array
        return cell_block(array, counts, repeats)

    from .supercell_levels import cell_levels, supercell_levels_setting

    levels = cell_levels(result, cell_problem, repeats)
    overrides = {
        "atoms": tuple(cell_problem.atoms),
        "grid": cell_grid,
        "electron_count": float(result.electron_count) / count,
        "eigenvalues": levels.eigenvalues,
        "occupations": levels.occupations,
        "k_points": levels.k_points,
        "k_weights": levels.weights,
        # the states of a k-point of the run, which --bands-from of the cell's input solves (off: the archive of
        # before, whose equal blocks give that count)
        "run_states_per_kpoint": levels.states if supercell_levels_setting() == "auto" else None,
        "energies": energies_per_cell(result.energies, count),
        "history": [
            replace(item, energies=energies_per_cell(item.energies, count))
            if hasattr(item, "__dataclass_fields__") and hasattr(item, "energies") else item
            for item in getattr(result, "history", ())
        ],
        "atomic_reference_correction": float(getattr(result, "atomic_reference_correction", 0.0)) / count,
        "all_electron_total": float(getattr(result, "all_electron_total", result.energies.total)) / count,
        "wavefunctions": None,
    }
    representations = getattr(result, "representations", None)
    if supercell_levels_setting() == "off" and representations is not None and (
        np.asarray(representations).size == np.asarray(host(result.eigenvalues)).size
    ):
        # PARSEC_KPOINT_SUPERCELL_LEVELS=off: the sectors of the supercell's levels, dealt with them, as before
        overrides["representations"] = np.asarray(representations)[
            _dealt_levels(np.asarray(host(result.eigenvalues), dtype=np.float64).reshape(-1), count)
        ]
    else:
        # the levels of a k-point carry no sector of the supercell (those of the k-points: one representation)
        overrides["representations"] = np.ones(levels.eigenvalues.size, dtype=np.int32)
    for name in ("density", "core_density", "ionic_potential", "hartree_potential", "xc_potential",
                 "input_effective_potential", "output_effective_potential", "next_effective_potential"):
        overrides[name] = block(getattr(result, name, None))
    potentials = getattr(result, "scf_potentials", None)
    if potentials:
        overrides["scf_potentials"] = {name: block(value) for name, value in potentials.items()}
    return _CellResult(result, overrides)


def cell_block(values, cell_counts, repeats) -> np.ndarray:
    """The first cell of a field on the supercell's grid (row-major, the cell's counts times ``repeats``)."""
    if hasattr(values, "get") and not isinstance(values, np.ndarray):
        values = values.get()
    array = np.asarray(values)
    counts = tuple(int(value) for value in cell_counts)
    supercounts = tuple(int(c) * int(m) for c, m in zip(counts, repeats))
    return np.ascontiguousarray(array.reshape(supercounts)[: counts[0], : counts[1], : counts[2]].reshape(-1))


__all__ = [
    "BLOCH_SWITCH",
    "DEFAULT_MOST_GPUS",
    "DEFAULT_TOLERANCE",
    "ESTIMATE_SAFETY",
    "GPUS_SWITCH",
    "MINIMUM_MESH_ENTRIES",
    "MINIMUM_SWITCH",
    "TOLERANCE_SWITCH",
    "TRIM_SWITCH",
    "TrimRoute",
    "cell_block",
    "cell_parsec_dat_view",
    "cell_result_view",
    "describe",
    "energies_per_cell",
    "kinetic_form_difference_estimate",
    "kpoint_supercell_gpus",
    "kpoint_supercell_minimum",
    "kpoint_supercell_setting",
    "kpoint_supercell_tolerance",
    "kpoints_of_supercell",
    "mesh_kpoints",
    "per_cell_lines",
    "process_gpu_count",
    "supercell_kpoints_setting",
    "supercell_of_trim_mesh",
    "trim_supercell_route",
    "validate_settings",
]
