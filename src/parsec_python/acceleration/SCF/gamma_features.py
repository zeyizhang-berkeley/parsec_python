"""TS, LDA+U, meta-GGA and libxc of bulk cells at the Gamma point in the symmetry sectors.

The route of the Gamma sectors (:func:`parsec_python.acceleration.driver.periodic_sector_refusal`) solves a bulk
orthorhombic cell at the Gamma point in the sectors of its point group.  Four inputs it refused until 2026-11-19 take
it under a switch of their own, each read in every run of the accelerated driver (an unknown value is an error
before the SCF):

``PARSEC_GAMMA_TS_SECTORS``
    the Tkatchenko--Scheffler correction (``Van_Der_Waals``).  It is formed after the converged loop from the
    density on the full grid of the cell (``SCF.single_point.dispersion_after_convergence``; the result of the
    sectors holds the density expanded from its orbits) and adds no potential, so the loop is that of the cell
    without it, in its weighted or abelian sectors; its forces are those of the same device sums.
``PARSEC_GAMMA_LDAU_SECTORS``
    LDA+U (``ldaplusu_u``, ``J = 0``).  The fifteen columns of every LDA+U atom are appended to the KB factor and
    reduced to every sector with it (``Hamiltonian/hubbard_device.py``); the occupation matrices are summed over
    the sector bases, ``B_G^T c = B^T psi``.  The term commutes with the operations ``r -> R r + t`` of the cell,
    which carry the d projectors of an atom onto those of its image, so the abelian sectors with their
    translations are exact, and so are the weighted sectors of a subgroup since 2026-11-21, whose occupation matrices
    are averaged over the full group (``PARSEC_LDAU_WEIGHTED_SECTORS``, ``hubbard_device.HubbardGroupAverage``;
    ``off``: the abelian group, ``driver._orbital_group_refusal``).
    Since 2026-11-21 the sectors carry the KB and LDA+U columns of one atom per orbit and the occupation matrices
    of the other atoms are those of the representatives turned with the d harmonics of the operations
    (``PARSEC_LDAU_PROJECTOR_ORBITS=auto``, ``hubbard_device.HubbardOrbits``; ``off``: the columns of every atom, as
    before).  A spin cell takes the spin sectors only under the opt-in ``PARSEC_LDAU_SPIN_SECTORS=auto`` or ``on``
    (:func:`ldau_spin_sector_setting`), each channel with its own potential matrices; under ``off``, the default, it
    keeps its route of before (the 1x1x1 k-point driver of the reference CLI).
``PARSEC_GAMMA_META_GGA_SECTORS``
    a meta-GGA (native SCAN or a libxc ``mgga_`` functional, which also needs ``PARSEC_GAMMA_LIBXC_SECTORS``):
    ``tau`` of the sector orbitals and the term ``-div(vtau grad)`` as the derivatives between the sectors
    (``Hamiltonian/meta_gga_device.py``; the derivative along a common eigenvector of the operations maps a sector
    onto the sector of the product of the characters, which the half-cell translations leave unchanged), the
    functional on the orbit fields (``V_xc/meta_gga_xc.py``).  Abelian sectors with their translations; since
    2026-11-23 the weighted sectors of the point group of the cell (``PARSEC_ORBITAL_GROUP_META_GGA_CELL``,
    ``Symmetry/weighted_inputs.py``: ``tau`` and ``g`` of the solved states averaged over the group).
``PARSEC_GAMMA_LIBXC_SECTORS``
    a libxc LDA or GGA: the device evaluators of :mod:`.V_xc.device_libxc` on the orbit fields (the stencil of the
    density gradient wrapped in the cell), or on the host on the expanded fields.  Abelian sectors with their
    translations (the weighted sectors of libxc are decided in ``driver._orbital_group_refusal``).

A meta-GGA and libxc in a spin-polarized cell take the route of the spin sectors (:mod:`.gamma_spin`), and so
does TS under ``on``; otherwise they keep the 1x1x1 k-point driver of the reference CLI, as before.  A TS input
carries ``Ignore_Symmetry`` (PARSEC error 159), which the command line reads as that requirement under ``auto`` or
``on`` where the sectors of a CuPy run take the cell (:func:`ts_ignore_symmetry_note`).

Values: ``auto`` takes the sectors where the basis of the cell (grid points times states) reaches
:data:`MINIMUM_ENTRIES` of its switch (2^27 for TS, 2^28 for libxc, 2^26 for LDA+U since 2026-11-21, 2^24 for a
meta-GGA since 2026-11-23, native or of libxc, ``PARSEC_GAMMA_META_GGA_FLOOR``: 2^27 before, and a libxc meta-GGA took
the floor of libxc too), below which the setup of the sectors costs more than they save; for TS only an unpolarized
cell (a spin cell was slower in the spin sectors than on the 1x1x1 k-point driver); :data:`MINIMUM_DEVICES` (from
2026-11-19 to 21 four CUDA devices for LDA+U) is empty; ``on`` takes them at every size; ``off`` keeps the full grid
of one device, as before.  Defaults (:data:`DEFAULTS`): ``auto`` for a meta-GGA, libxc, LDA+U and TS (TS since 2026-11-21: the
``Ignore_Symmetry`` of a TS input is read as PARSEC 1.5's requirement of the correction, error 159, not as a request
for the full grid; ``off`` restores the former default).  The energies are those of the full grid to the
convergence of the SCF (forces of a meta-GGA, of libxc and, under ``PARSEC_LDAU_SECTOR_FORCES=auto``, of LDA+U too;
the forces of TS inputs, which run on the Fortran-derived routines, read the full grid:
``acceleration.cli.periodic_sector_forces_refusal``).  A hybrid with
one of them keeps the refusal (the Fock loop of the sectors, :mod:`.gamma_hybrid_route`, was validated with native
semilocal parts and no LDA+U), and so do the callers that run several ranks or fold a k-point mesh into a supercell
(``periodic_sector_refusal(..., features=False)``, the default): the MPI full-SCF runner and the supercell of
:mod:`parsec_python.SCF.kpoint_supercell`.
"""

from __future__ import annotations

import os

TS_SWITCH = "PARSEC_GAMMA_TS_SECTORS"
LDAU_SWITCH = "PARSEC_GAMMA_LDAU_SECTORS"
META_GGA_SWITCH = "PARSEC_GAMMA_META_GGA_SECTORS"
LIBXC_SWITCH = "PARSEC_GAMMA_LIBXC_SECTORS"

#: The default of each switch (since 2026-11-19, validated on A100 against the full grid, see :data:`MINIMUM_ENTRIES`):
#: ``auto`` for a meta-GGA and libxc, and since 2026-11-21 for LDA+U (its sectors carry one atom per orbit and take
#: the weighted sectors of the point group: ``Hamiltonian/hubbard_device.py``; ``off`` from 2026-11-19 to 21, when
#: every sector carried the columns of every atom and one GPU was slower) and for TS (``off`` before): it reads the
#: ``Ignore_Symmetry`` that PARSEC 1.5 demands of every TS input (error 159) as that requirement rather than as a
#: request for the full grid, and takes the sectors of an unpolarized cell from :data:`MINIMUM_ENTRIES` on (the same
#: energies as the full grid, faster on every cell measured above the floor: README, ``PARSEC_GAMMA_TS_SECTORS``).
DEFAULTS = {
    TS_SWITCH: "auto",
    LDAU_SWITCH: "auto",
    META_GGA_SWITCH: "auto",
    LIBXC_SWITCH: "auto",
}

#: The smallest basis (grid points of the cell times its states) that ``auto`` takes into the sectors, per switch:
#: below it the setup of the many small abelian sectors with their translations costs more than the loop saves.  TS
#: and meta-GGAs save more per iteration (the weighted sectors of TS; the host functional of SCAN on the orbits) than
#: LDA+U and libxc, whose full grid has cheap device terms.  One node of four A100, one process, first runs, SCF
#: (total) seconds, full grid -> sectors: Si64 48^3 x 153 (1.7e7) libxc PBE 2.2 -> 7.9 (4.4 -> 15.9), SCAN 14.2 ->
#: 17.2 (16.6 -> 25.8), PBE + TS 5.2 -> 5.1 (7.1 -> 11.4); Ag108 48^3 x 778 (8.6e7) LDA+U 13.4 -> 17.9 (23.8 ->
#: 33.7); Si216 72^3 x 502 (1.9e8) libxc PBE 16.2 -> 12.8 (19.0 -> 20.9; again 17.8 -> 20.6), SCAN 254.3 -> 19.3
#: (257.7 -> 28.8), PBE + TS 30.9 -> 18.9 (34.0 -> 25.5); Ag108 70^3 x 778 (2.67e8, below the floor, sectors under
#: ``on``) LDA+U 41.6 -> 23.5 (67.8 -> 55.2); Si216 84^3 x 502 (3.0e8) libxc PBE 26.1 -> 14.1 (29.1 -> 22.2);
#: Ag256 64^3 x 1836 (4.8e8) LDA+U 70.3 -> 46.9 (115.6 -> 103.3); Si512 96^3 x 1183 (1.0e9) libxc PBE 96.1 -> 30.6
#: (99.9 -> 41.2).  ``on`` takes every size.
#:
#: LDA+U since 2026-11-21 (one atom per orbit, the weighted sectors of the point group with the face-centring
#: translations; SCF (total) seconds on one, two and four A100, first runs): fcc Ag32 at 46^3 x 234 (2.3e7) 3.4
#: (5.8) -> 3.5 (11.6), 4.0 (12.2), 4.3 (13.6), the setup of the sectors 2.4 -> 8.4 s; SrTiO3 40 atoms at 54^3 x 118
#: (1.9e7) 4.3 (6.7) -> 5.1 (13.3), 5.8 (14.4), 5.9 (15.3); Ag108 48^3 x 778 (8.6e7) 11.6 (14.1) -> 6.3 (9.4),
#: 5.6 (10.1) on four; SrTiO3 135 atoms 60^3 x 380 (8.2e7) 8.8 (11.6) -> 4.5 (7.0), 5.3 (9.0) on four; SrTiO3 320
#: atoms 84^3 x 891 (5.3e8) 48.2 (52.3) -> 8.7 (18.1), 9.9 (20.7) on four; Ag256 64^3 x 1836 (4.8e8) 61.9 (65.2)
#: -> 9.9 (18.5), 11.6 (21.5) on four; the energies to 1e-8 Ry, the occupation matrices within 1.5e-7.
#:
#: A meta-GGA since 2026-11-23, in the weighted sectors of the point group of the cell
#: (``PARSEC_ORBITAL_GROUP_META_GGA_CELL``): 2^24 entries, the floor of every unpolarized cell
#: (``PARSEC_PERIODIC_SECTOR_FLOOR``), for native SCAN and for the libxc ``mgga_`` functionals, which no longer take the
#: floor of libxc (2^28, measured for its LDA and GGA, whose full grid has cheap device terms);
#: ``PARSEC_GAMMA_META_GGA_FLOOR`` (:func:`meta_gga_floor_setting`; ``former``: 2^27 and the floor of libxc).  First
#: runs, FHI Si, criterion 1e-6 Ry, total seconds, the full grid (a spin cell: the 1x1x1 k-point driver) -> the sectors,
#: one A100 / one process on four: Si64 at 48^3 x 160 (1.8e7) 32.1 -> 23.7 / 32.2 -> 24.3; the same with an atom moved
#: (the abelian sectors of C2v) 31.7 -> 20.2 / 31.7 -> 20.0; spin Si64 at 48^3 x 160 73.7 -> 33.7 / 40.8 -> 34.1, at
#: 64^3 x 144 (3.8e7) - / 99.9 -> 33.1; Si64 at 64^3 51.8 -> 25.5 / 50.7 -> 26.0, at 82^3 (8.8e7) 87.9 -> 25.7;
#: Si216 at 64^3 x 486 (1.3e8) 110.1 -> 25.7 / 109.4 -> 26.9; libxc r2SCAN Si64 at 64^3 45.3 -> 18.5 / 45.1 -> 18.8,
#: Si216 at 82^3 x 486 (2.7e8) 193.8 -> 20.3 (GPU peak 25.7 -> 0.8 GiB); with forces (Si64, Td with the translations,
#: 1e-8 Ry) 37.3 -> 25.4, the forces within 1.4e-9 Ry/bohr.  Below 2^24 the sectors gain little or nothing on the
#: small cells (Si8, Si16: 0.9 to 1.07 of the full grid's time on four A100), a spin Si64 at 40^3 (9.2e6) was slower
#: on four (24.9 -> 28.5) and at 32^3 did not converge in 200 steps in the sectors (the k-point driver did in 116).
MINIMUM_ENTRIES = {
    TS_SWITCH: 1 << 27,
    LDAU_SWITCH: 1 << 26,
    META_GGA_SWITCH: 1 << 24,
    LIBXC_SWITCH: 1 << 28,
}

#: ``PARSEC_GAMMA_META_GGA_FLOOR``: the floor of a meta-GGA under ``PARSEC_GAMMA_META_GGA_SECTORS=auto``.
META_GGA_FLOOR_SWITCH = "PARSEC_GAMMA_META_GGA_FLOOR"
#: The floor of a meta-GGA from 2026-11-19 to 23 (``former``), which a libxc meta-GGA took with that of libxc.
META_GGA_FORMER_FLOOR = 1 << 27

#: The fewest CUDA devices of the process (``PARSEC_CUPY_DEVICES``) with which ``auto`` takes a switch into the
#: sectors; none since 2026-11-21.  From 2026-11-19 LDA+U needed four: its sectors carried the KB and LDA+U columns of
#: every atom, Ag256 fcc 64^3 x 1836 (4.8e8 entries) total 115.6 -> 103.3 s on four A100 but 114.4 -> 170.9 s on one.
#: With one atom per orbit (``PARSEC_LDAU_PROJECTOR_ORBITS``) one GPU is the fastest: 106.5 -> 75.1 s on one, 80.8 on
#: two, 84.2 on four (the abelian sectors), and the weighted sectors are faster still on every count measured.
MINIMUM_DEVICES: dict[str, int] = {}

#: The reasons of the refusals, as the route of the sectors gave them before the switches.
TS_REASON = "the Tkatchenko-Scheffler dispersion correction"
LDAU_REASON = "LDA+U"
META_GGA_REASON = "a meta-GGA functional"
LIBXC_REASON = "a libxc functional"


def gamma_feature_setting(name: str) -> str:
    """The value of one of the four switches: ``auto``, ``on`` or ``off`` (``0``, ``false``, ``no`` mean ``off``)."""

    if name not in DEFAULTS:
        raise KeyError(name)
    default = DEFAULTS[name]
    value = os.environ.get(name, default).strip().lower() or default
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{name} must be auto, on or off, not {value!r}")
    return value


def gamma_feature_settings() -> dict[str, str]:
    """Every switch of this module, read (and checked) at once (``PARSEC_GAMMA_META_GGA_FLOOR`` too)."""

    meta_gga_floor_setting()
    return {name: gamma_feature_setting(name) for name in DEFAULTS}


def meta_gga_floor_setting() -> tuple[int, bool]:
    """``PARSEC_GAMMA_META_GGA_FLOOR``: ``(floor, with the floor of libxc)`` of a meta-GGA under
    ``PARSEC_GAMMA_META_GGA_SECTORS=auto``.

    ``auto`` (the default since 2026-11-23): :data:`MINIMUM_ENTRIES` of the meta-GGA (2^24 grid points x states) for
    native SCAN and for a libxc ``mgga_`` functional alike, whose ``PARSEC_GAMMA_LIBXC_SECTORS`` must not be ``off``
    but whose floor (2^28) it does not take; ``former``: the rule from 2026-11-19 to 23, 2^27 and, for a libxc
    meta-GGA, the floor of libxc as well (the routes of before, bit for bit); an integer ``>= 0``: that floor for
    every meta-GGA.  Below ``PARSEC_PERIODIC_SECTOR_FLOOR`` an unpolarized cell keeps the full grid whatever this
    floor.
    """

    raw = os.environ.get(META_GGA_FLOOR_SWITCH, "auto").strip().lower() or "auto"
    if raw == "auto":
        return int(MINIMUM_ENTRIES[META_GGA_SWITCH]), False
    if raw == "former":
        return int(META_GGA_FORMER_FLOOR), True
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{META_GGA_FLOOR_SWITCH} must be auto, former or a number of entries >= 0, not {raw!r}")
    return value, False


def _floor_entries(switch: str) -> int:
    """The floor of ``switch`` under ``auto`` (that of a meta-GGA from :func:`meta_gga_floor_setting`)."""

    if switch == META_GGA_SWITCH:
        return meta_gga_floor_setting()[0]
    return int(MINIMUM_ENTRIES[switch])


#: Whether the forces and relaxations of a Gamma cell with LDA+U whose SCF takes the sectors keep them (``auto``)
#: or read the full grid (``off``).
LDAU_FORCES_SWITCH = "PARSEC_LDAU_SECTOR_FORCES"
LDAU_FORCES_DEFAULT = "auto"


def ldau_sector_forces_setting() -> str:
    """``PARSEC_LDAU_SECTOR_FORCES``: ``auto`` or ``off`` (``0``, ``false``, ``no`` mean ``off``).

    ``auto``: the forces and the ``bfgs``/``simple`` relaxations of a bulk cell at the Gamma point with LDA+U (an
    input of the Fortran-derived routines) keep the sectors where :func:`parsec_python.acceleration.cli.
    periodic_sector_forces_refusal` takes the forces of those routines (``PARSEC_FORTRAN_SECTOR_FORCES=auto``): the
    nonlocal KB term in the sectors of one atom per orbit completed over the group, the LDA+U term of every atom from
    the sector bases with the gradients of its projectors (``Hamiltonian/hubbard_device.DeviceHubbard.forces``),
    the total and the moves averaged over the group.  ``off`` keeps the full grid for them, as before.
    """

    value = os.environ.get(LDAU_FORCES_SWITCH, LDAU_FORCES_DEFAULT).strip().lower() or LDAU_FORCES_DEFAULT
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "off"):
        raise ValueError(f"{LDAU_FORCES_SWITCH} must be auto or off, not {value!r}")
    return value


#: Whether a spin-polarized LDA+U run takes the spin sectors: ``off`` (the default) neither a cell nor a cluster (the
#: refusal of before); opt-in, ``auto`` a Gamma cell, ``on`` a cluster too.
LDAU_SPIN_SWITCH = "PARSEC_LDAU_SPIN_SECTORS"
LDAU_SPIN_DEFAULT = "off"


def ldau_spin_sector_setting() -> str:
    """``PARSEC_LDAU_SPIN_SECTORS``: ``off`` (the default), ``auto`` or ``on`` (``0``, ``false``, ``no`` mean ``off``).

    ``off`` keeps the routes of before: a spin-polarized bulk cell at the Gamma point with LDA+U on the 1x1x1
    k-point driver of the reference CLI, a spin cluster with LDA+U on the full grid of the driver.  The opt-in values
    let the two channels take the spin sectors, each channel with its own potential matrices written into the shared
    sector operators before its solve (``Hamiltonian/hubbard_device.py``), with the guard of :mod:`.spin_sectors`:
    ``auto`` a bulk cell at the Gamma point where its unpolarized twin takes the sectors (the LDA+U switch
    :data:`LDAU_SWITCH` of this module, then the spin route of :mod:`.gamma_spin`), ``on`` also a cluster
    (``Spin_Symmetry_Sectors``, ``driver._spin_sector_refusal``).  Opt-in since 2026-11-21 (``auto`` was the default
    for that day): the sectors can end, without a note of the guard, in another magnetic state than the k-point
    driver (bcc Cr16 AFM, U = 2 eV, seeds +/-0.02 on 100^3 points: the non-magnetic -303.52817316 Ry, where the
    k-point driver stays magnetic near -306.549 Ry; README, the row of this switch).
    """

    value = os.environ.get(LDAU_SPIN_SWITCH, LDAU_SPIN_DEFAULT).strip().lower() or LDAU_SPIN_DEFAULT
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{LDAU_SPIN_SWITCH} must be auto, on or off, not {value!r}")
    return value


#: Whether a spin-polarized LDA+U cell below the floor of :data:`LDAU_SWITCH` leaves the 1x1x1 k-point driver of the
#: reference CLI for the full grid of the accelerated driver (:func:`ldau_spin_full_grid`).  Opt-in: the route is faster
#: but can converge to another magnetic state.
LDAU_SPIN_FULL_GRID_SWITCH = "PARSEC_LDAU_SPIN_FULL_GRID"
LDAU_SPIN_FULL_GRID_DEFAULT = "off"


def ldau_spin_full_grid_setting() -> str:
    """``PARSEC_LDAU_SPIN_FULL_GRID``: ``off`` (the default), ``auto`` or ``on`` (``0``, ``false``, ``no`` mean ``off``).

    ``off`` keeps a spin-polarized bulk cell at the Gamma point with LDA+U below the floor of the sectors
    (:data:`MINIMUM_ENTRIES`) on the 1x1x1 k-point driver of the reference CLI.  ``auto`` and ``on`` are opt-in: a
    single point of such a cell is solved on the full grid of one device of the accelerated driver
    (:func:`ldau_spin_full_grid` says for which inputs).  That is faster, but the two drivers do not follow the same
    SCF path and can end in different magnetic states, which nothing in the input tells apart (A100, k-point driver
    -> full grid: bcc Cr16 AFM, U = 2 eV, seeds +/-0.1 to 0.3, -306.35013065 Ry at moment 0 -> -306.38211140 at 5.41;
    seeds +/-0.05, -306.75421598 -> not converged in 200 iterations; MnO AFM-I 8 atoms, ``Net_Charges`` +1,
    -242.46453788 at moment 0 -> -242.47139075 at 1.00).
    """

    value = (
        os.environ.get(LDAU_SPIN_FULL_GRID_SWITCH, LDAU_SPIN_FULL_GRID_DEFAULT).strip().lower()
        or LDAU_SPIN_FULL_GRID_DEFAULT
    )
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{LDAU_SPIN_FULL_GRID_SWITCH} must be auto, on or off, not {value!r}")
    return value


def _shares_spin_start(problem) -> bool:
    """Whether the k-point driver starts this spin input from the start of the accelerated driver.

    The accelerated driver takes the start of ``initchrg.f90`` for every input, each species with its own
    ``Initial_Spin_Polarization``.  The k-point loop takes it under ``PARSEC_SPIN_START=initchrg`` and, under
    ``auto``, where the species start with values of both signs (an antiferromagnetic or broken-symmetry start of two
    labels); elsewhere it starts from one polarization fraction of the whole cell, and ``auto`` of
    ``PARSEC_LDAU_SPIN_FULL_GRID`` leaves those inputs on the k-point driver (MnO in 64 atoms with one value for every
    species reached two states 4.9e-3 Ry apart, both at 70 Bohr magnetons).  The same start does not make the same
    path: the opt-in note of :func:`ldau_spin_full_grid_setting` lists inputs with this start that end elsewhere.
    """

    from parsec_python.SCF.spin_polarized import spin_start_setting

    setting = spin_start_setting()
    if setting != "auto":
        return setting == "initchrg"
    specifications = problem.pseudopotentials
    values = [float(specifications[atom.symbol].initial_spin_polarization) for atom in problem.atoms]
    return bool(values) and min(values) < 0.0 < max(values)


def _has_hubbard(problem) -> bool:
    return any(getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values())


def basis_entries(problem) -> int | None:
    """Grid points of the cell times the states of ``problem`` (``None`` where they cannot be found here)."""

    from parsec_python.Grid.pbc import build_periodic_grid

    from .gamma_hybrid_route import _states

    try:
        rows = int(build_periodic_grid(problem.periodic_cell, problem.grid).size)
        states = _states(problem)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return rows * states


def _below_floor(problem, switch: str) -> bool:
    entries = basis_entries(problem)
    return entries is None or entries < _floor_entries(switch)


def process_device_count() -> int:
    """The CUDA devices that the sectors of this process spread over (``PARSEC_CUPY_DEVICES``), 0 without CuPy or
    under ``PARSEC_DEVICE=cpu`` (which never probes a GPU).

    Asked only in one process (``features``), so no MPI rank takes a decision of its own from it.
    """

    from ..backends.cupy import cupy_available
    from ..resources import device_policy

    if device_policy() == "cpu" or not cupy_available():
        return 0
    from ..backends.cupy import require_cupy
    from ..Eigensolvers.symmetry import selected_device_ids

    cp, _ = require_cupy()
    return len(selected_device_ids(int(cp.cuda.Device().id)))


def _floor_refusal(reason: str, switch: str) -> str:
    """The refusal of a cell whose basis is below the floor of ``switch`` under ``auto``."""

    return f"{reason} (a basis below {_floor_entries(switch)} grid points x states under {switch}=auto; on takes it)"


def _refused(
    problem,
    switch: str,
    reason: str,
    *,
    features: bool,
    refused: bool = False,
    auto_refusal: str | None = None,
    floor: bool = True,
) -> str | None:
    """``reason`` where the switch keeps the feature off the sectors, with why under ``auto``; ``None`` where it
    takes them.  ``refused``: a combination that no value takes (a hybrid; LDA+U of a spin cell).
    ``auto_refusal``: why ``auto`` keeps the route of before for this cell whatever its size (``on`` takes it).
    ``floor`` false: ``auto`` applies no floor of this switch (the libxc switch of a meta-GGA, whose floor is that
    of the meta-GGA, :func:`meta_gga_floor_setting`)."""

    if not features or refused:
        return reason
    value = gamma_feature_setting(switch)
    if value == "off":
        return reason
    if value == "auto":
        if auto_refusal is not None:
            return f"{reason} ({auto_refusal} under {switch}=auto; on takes it)"
        if floor and _below_floor(problem, switch):
            return _floor_refusal(reason, switch)
        devices = MINIMUM_DEVICES.get(switch)
        if devices is not None and process_device_count() < devices:
            return f"{reason} (fewer than {devices} CUDA devices in this process under {switch}=auto; on takes it)"
    return None


def meta_gga_or_libxc_refusal(problem, *, features: bool) -> str | None:
    """The refusal of a meta-GGA or libxc functional by the route of the Gamma sectors, or ``None``.

    ``features`` false (a caller that runs several ranks or folds a mesh): refused as before.  Otherwise each takes
    the sectors under its switch (``on``; ``auto`` from :data:`MINIMUM_ENTRIES` on, a meta-GGA from the floor of
    ``PARSEC_GAMMA_META_GGA_FLOOR``) where the functional is not a hybrid; a libxc meta-GGA needs both switches, and
    since 2026-11-23 takes the floor of the meta-GGA alone (``PARSEC_GAMMA_META_GGA_FLOOR=former``: that of libxc as
    well, as before).
    """

    from parsec_python.V_xc.libxc import is_libxc_functional

    scf = problem.scf
    hybrid = bool(getattr(scf, "is_hybrid", False))
    meta = bool(getattr(scf, "is_meta_gga", False))
    if meta:
        found = _refused(problem, META_GGA_SWITCH, META_GGA_REASON, features=features, refused=hybrid)
        if found is not None:
            return found
    if is_libxc_functional(getattr(scf, "xc_functional", "ca")):
        # (a libxc meta-GGA: the floor of the meta-GGA alone, unless PARSEC_GAMMA_META_GGA_FLOOR=former)
        floor = not meta or meta_gga_floor_setting()[1]
        return _refused(problem, LIBXC_SWITCH, LIBXC_REASON, features=features, refused=hybrid, floor=floor)
    return None


def correction_refusal(problem, *, features: bool) -> str | None:
    """The refusal of LDA+U or of the TS correction by the route of the Gamma sectors, or ``None``.

    LDA+U stays refused with a hybrid, and for a spin-polarized cell under ``PARSEC_LDAU_SPIN_SECTORS=off`` (the
    default): under the opt-in ``auto`` or ``on`` a spin cell takes the spin sectors where its switch
    (:data:`LDAU_SWITCH`, from 2^26 grid points x states under ``auto``) takes the unpolarized one.  TS of a
    spin-polarized cell takes the spin sectors only under ``on``: under ``auto`` it keeps the 1x1x1 k-point driver,
    which was faster (Si216 spin PBE + TS, 72^3 x 502, four A100, first runs, in both orders: SCF 27.3 -> 39.8 s and
    27.1 -> 40.2 s, process wall 46.9 -> 50.5 s and 46.1 -> 51.3 s, the same energy).
    """

    scf = problem.scf
    hybrid = bool(getattr(scf, "is_hybrid", False))
    spin = bool(getattr(scf, "spin_polarized", False))
    if _has_hubbard(problem):
        spin_refused = spin and ldau_spin_sector_setting() == "off"
        found = _refused(problem, LDAU_SWITCH, LDAU_REASON, features=features, refused=hybrid or spin_refused)
        if found is not None:
            return found
    if getattr(scf, "van_der_waals", False):
        return _refused(
            problem,
            TS_SWITCH,
            TS_REASON,
            features=features,
            refused=hybrid,
            auto_refusal="a spin-polarized cell, which keeps the 1x1x1 k-point driver," if spin else None,
        )
    return None


def ldau_spin_full_grid(problem, refusal: str | None) -> bool:
    """Whether a spin LDA+U cell that the sectors refuse takes the full grid of the accelerated driver.

    ``refusal`` is what :func:`parsec_python.acceleration.driver.periodic_sector_refusal` said of ``problem`` in one
    process.  Only a cell below the floor of LDA+U (:data:`MINIMUM_ENTRIES`) with ``PARSEC_GAMMA_LDAU_SECTORS=auto``
    qualifies, refused for that floor or for ``PARSEC_LDAU_SPIN_SECTORS=off`` (the default), so every other reason
    (``PARSEC_GAMMA_LDAU_SECTORS`` at ``on`` or ``off``, k-points, an oblique cell, a wire or slab) keeps the k-point
    driver, as do ``J != 0`` (which the accelerated driver refuses), a hybrid, TS, a meta-GGA and libxc beside LDA+U.
    The
    default ``PARSEC_LDAU_SPIN_FULL_GRID=off`` takes none; the opt-in ``auto`` takes the inputs that both drivers start
    alike (:func:`_shares_spin_start`), ``on`` the others too.
    """

    from parsec_python.V_xc.libxc import is_libxc_functional

    setting = ldau_spin_full_grid_setting()
    if setting == "off" or refusal is None:
        return False
    if refusal != _floor_refusal(LDAU_REASON, LDAU_SWITCH) and not (
        refusal == LDAU_REASON
        and ldau_spin_sector_setting() == "off"
        and gamma_feature_setting(LDAU_SWITCH) == "auto"
        and _below_floor(problem, LDAU_SWITCH)
    ):
        return False
    scf = problem.scf
    if (
        not getattr(scf, "spin_polarized", False)
        or getattr(scf, "is_hybrid", False)
        or getattr(scf, "is_meta_gga", False)
        or getattr(scf, "van_der_waals", False)
        or is_libxc_functional(getattr(scf, "xc_functional", "ca"))
    ):
        return False
    species = getattr(problem, "pseudopotentials", {}).values()
    if any(getattr(item, "has_hubbard", False) and float(item.hubbard_j) != 0.0 for item in species):
        return False
    return setting == "on" or _shares_spin_start(problem)


def ts_ignore_symmetry_note(translation, *, backend_is_cupy) -> str | None:
    """The NOTE of a TS cell whose ``Ignore_Symmetry`` does not keep it off the sectors, or ``None``.

    PARSEC 1.5 refuses the TS correction unless symmetry is switched off (``Ignore_Symmetry: true``, error 159:
    its Hirshfeld sums read a density it would otherwise hold on the irreducible wedge), so every TS input says
    so.  Here the correction reads the density on the full grid of the cell whatever the route of the loop, and the
    sectors give the energy of the full grid; under ``PARSEC_GAMMA_TS_SECTORS`` ``auto`` (the default since
    2026-11-21) or ``on`` the command line therefore reads that keyword of a TS input as PARSEC's requirement, not as a request
    for the full grid, where the route of the sectors takes the cell: the point group of the cell is looked for
    (``PARSEC_PERIODIC_SYMMETRY=auto``), the backend resolves to CuPy (``backend_is_cupy()``) and
    :func:`parsec_python.acceleration.driver.periodic_sector_refusal` takes it in one process (under ``auto`` an
    unpolarized cell from 2^27 grid points times states on, :data:`MINIMUM_ENTRIES`).  ``--symmetry off``,
    ``PARSEC_PERIODIC_SYMMETRY=off``, another backend or ``PARSEC_GAMMA_TS_SECTORS=off`` keep the full grid.
    ``None`` for every other input (and in those cases), whose ``Ignore_Symmetry`` keeps its meaning.
    """

    problem = translation.problem
    if (
        problem.periodic_cell is None
        or not getattr(problem.scf, "van_der_waals", False)
        or not getattr(translation, "ignore_symmetry", False)
        or gamma_feature_setting(TS_SWITCH) == "off"
    ):
        return None
    from ..Symmetry import periodic_symmetry_setting

    if periodic_symmetry_setting() == "off" or not backend_is_cupy():
        return None
    from ..driver import periodic_sector_refusal

    if periodic_sector_refusal(problem, features=True) is not None:
        return None
    return (
        "NOTE: Ignore_Symmetry of this TS input (PARSEC 1.5 requires it of Van_Der_Waals, error 159) does not keep "
        "the cell off the sectors of its point group: the TS correction reads the density on the full grid of the "
        f"cell ({TS_SWITCH}={gamma_feature_setting(TS_SWITCH)}; --symmetry off or {TS_SWITCH}=off keep the full grid)"
    )


def taken_features(problem) -> list[str]:
    """The switches of this module that bring ``problem`` into the sectors (for the report), in a fixed order."""

    from parsec_python.V_xc.libxc import is_libxc_functional

    scf = problem.scf
    names = []
    if getattr(scf, "van_der_waals", False):
        names.append(TS_SWITCH)
    if _has_hubbard(problem):
        names.append(LDAU_SWITCH)
    if getattr(scf, "is_meta_gga", False):
        names.append(META_GGA_SWITCH)
    if is_libxc_functional(getattr(scf, "xc_functional", "ca")):
        names.append(LIBXC_SWITCH)
    return names


__all__ = [
    "DEFAULTS",
    "LDAU_FORCES_DEFAULT",
    "LDAU_FORCES_SWITCH",
    "LDAU_SPIN_DEFAULT",
    "LDAU_SPIN_FULL_GRID_DEFAULT",
    "LDAU_SPIN_FULL_GRID_SWITCH",
    "LDAU_SPIN_SWITCH",
    "LDAU_SWITCH",
    "LIBXC_SWITCH",
    "META_GGA_FLOOR_SWITCH",
    "META_GGA_FORMER_FLOOR",
    "META_GGA_SWITCH",
    "meta_gga_floor_setting",
    "TS_SWITCH",
    "correction_refusal",
    "gamma_feature_setting",
    "gamma_feature_settings",
    "ldau_sector_forces_setting",
    "ldau_spin_full_grid",
    "ldau_spin_full_grid_setting",
    "ldau_spin_sector_setting",
    "meta_gga_or_libxc_refusal",
    "taken_features",
    "ts_ignore_symmetry_note",
    "MINIMUM_DEVICES",
    "MINIMUM_ENTRIES",
    "basis_entries",
    "process_device_count",
]
