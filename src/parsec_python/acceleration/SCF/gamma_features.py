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
    translations are exact (the weighted sectors are refused, as on clusters: ``driver._orbital_group_refusal``),
    and every atom keeps its projectors (no projector orbits with LDA+U).  Unpolarized only: the spin sectors carry
    no LDA+U (``driver._spin_sector_refusal``), so a spin cell with LDA+U keeps its route.
``PARSEC_GAMMA_META_GGA_SECTORS``
    a meta-GGA (native SCAN or a libxc ``mgga_`` functional, which also needs ``PARSEC_GAMMA_LIBXC_SECTORS``):
    ``tau`` of the sector orbitals and the term ``-div(vtau grad)`` as the derivatives between the sectors
    (``Hamiltonian/meta_gga_device.py``; the derivative along a common eigenvector of the operations maps a sector
    onto the sector of the product of the characters, which the half-cell translations leave unchanged), the
    functional on the orbit fields (``V_xc/meta_gga_xc.py``).  Abelian sectors with their translations.
``PARSEC_GAMMA_LIBXC_SECTORS``
    a libxc LDA or GGA: the device evaluators of :mod:`.V_xc.device_libxc` on the orbit fields (the stencil of the
    density gradient wrapped in the cell), or on the host on the expanded fields.  Abelian sectors with their
    translations (the weighted sectors of libxc are decided in ``driver._orbital_group_refusal``).

A meta-GGA and libxc in a spin-polarized cell take the route of the spin sectors (:mod:`.gamma_spin`), and so
does TS under ``on``; otherwise they keep the 1x1x1 k-point driver of the reference CLI, as before.  A TS input
carries ``Ignore_Symmetry`` (PARSEC error 159), which the command line reads as that requirement under ``auto`` or
``on`` where the sectors of a CuPy run take the cell (:func:`ts_ignore_symmetry_note`).

Values: ``auto`` takes the sectors where the basis of the cell (grid points times states) reaches
:data:`MINIMUM_ENTRIES` of its switch (2^27 for TS and a meta-GGA, 2^28 for LDA+U and libxc), below which the many
small abelian sectors with their translations cost more than they save; for TS only an unpolarized cell (a spin
cell was slower in the spin sectors than on the 1x1x1 k-point driver) and for LDA+U only a process with
:data:`MINIMUM_DEVICES` CUDA devices (on one GPU the sectors were slower); ``on`` takes them at every size;
``off`` keeps the full grid of one device, as before.  Defaults (:data:`DEFAULTS`): ``auto`` for a meta-GGA and
libxc, ``off`` for TS (the reading of its ``Ignore_Symmetry`` departs from PARSEC 1.5, so it is
opt-in) and for LDA+U (measured faster only on four GPUs).  The energies are those of the full grid to the
convergence of the SCF (forces of a meta-GGA and of libxc too; the forces of TS and LDA+U inputs, which run on the
Fortran-derived routines, read the full grid: ``acceleration.cli.periodic_sector_forces_refusal``).  A hybrid with
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
#: ``auto`` for a meta-GGA and libxc.  ``off`` for TS, whose ``auto`` reads the ``Ignore_Symmetry`` of a TS input as
#: PARSEC's requirement rather than as a request for the full grid (a departure from PARSEC 1.5, hence opt-in),
#: and for LDA+U, whose sectors were slower on one GPU (:data:`MINIMUM_DEVICES`).
DEFAULTS = {
    TS_SWITCH: "off",
    LDAU_SWITCH: "off",
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
MINIMUM_ENTRIES = {
    TS_SWITCH: 1 << 27,
    LDAU_SWITCH: 1 << 28,
    META_GGA_SWITCH: 1 << 27,
    LIBXC_SWITCH: 1 << 28,
}

#: The fewest CUDA devices of the process (``PARSEC_CUPY_DEVICES``) with which ``auto`` takes a switch into the
#: sectors.  LDA+U carries the KB and LDA+U columns of every atom into every sector (no projector orbits), which
#: four devices share and one carries alone: Ag256 fcc 64^3 x 1836 (4.8e8 entries), first runs, total seconds full
#: grid -> sectors, 115.6 -> 103.3 on four A100 but 114.4 -> 170.9 with one visible A100 (SCF 69.9 -> 115.5); TS and
#: libxc stayed faster on one A100 (Si216 PBE + TS 33.9 -> 23.8, libxc Si216 84^3 29.0 -> 18.7).
MINIMUM_DEVICES = {
    LDAU_SWITCH: 4,
}

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
    """Every switch of this module, read (and checked) at once."""

    return {name: gamma_feature_setting(name) for name in DEFAULTS}


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
    return entries is None or entries < MINIMUM_ENTRIES[switch]


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


def _refused(
    problem, switch: str, reason: str, *, features: bool, refused: bool = False, auto_refusal: str | None = None
) -> str | None:
    """``reason`` where the switch keeps the feature off the sectors, with why under ``auto``; ``None`` where it
    takes them.  ``refused``: a combination that no value takes (a hybrid; LDA+U of a spin cell).
    ``auto_refusal``: why ``auto`` keeps the route of before for this cell whatever its size (``on`` takes it)."""

    if not features or refused:
        return reason
    value = gamma_feature_setting(switch)
    if value == "off":
        return reason
    if value == "auto":
        if auto_refusal is not None:
            return f"{reason} ({auto_refusal} under {switch}=auto; on takes it)"
        if _below_floor(problem, switch):
            return (
                f"{reason} (a basis below {MINIMUM_ENTRIES[switch]} grid points x states under {switch}=auto; "
                "on takes it)"
            )
        devices = MINIMUM_DEVICES.get(switch)
        if devices is not None and process_device_count() < devices:
            return f"{reason} (fewer than {devices} CUDA devices in this process under {switch}=auto; on takes it)"
    return None


def meta_gga_or_libxc_refusal(problem, *, features: bool) -> str | None:
    """The refusal of a meta-GGA or libxc functional by the route of the Gamma sectors, or ``None``.

    ``features`` false (a caller that runs several ranks or folds a mesh): refused as before.  Otherwise each takes
    the sectors under its switch (``on``; ``auto`` from :data:`MINIMUM_ENTRIES` on) where the functional is not a
    hybrid.
    """

    from parsec_python.V_xc.libxc import is_libxc_functional

    scf = problem.scf
    hybrid = bool(getattr(scf, "is_hybrid", False))
    if getattr(scf, "is_meta_gga", False):
        found = _refused(problem, META_GGA_SWITCH, META_GGA_REASON, features=features, refused=hybrid)
        if found is not None:
            return found
    if is_libxc_functional(getattr(scf, "xc_functional", "ca")):
        return _refused(problem, LIBXC_SWITCH, LIBXC_REASON, features=features, refused=hybrid)
    return None


def correction_refusal(problem, *, features: bool) -> str | None:
    """The refusal of LDA+U or of the TS correction by the route of the Gamma sectors, or ``None``.

    LDA+U also stays refused for a spin-polarized cell (the spin sectors carry no LDA+U) and with a hybrid.  TS of a
    spin-polarized cell takes the spin sectors only under ``on``: under ``auto`` it keeps the 1x1x1 k-point driver,
    which was faster (Si216 spin PBE + TS, 72^3 x 502, four A100, first runs, in both orders: SCF 27.3 -> 39.8 s and
    27.1 -> 40.2 s, process wall 46.9 -> 50.5 s and 46.1 -> 51.3 s, the same energy).
    """

    scf = problem.scf
    hybrid = bool(getattr(scf, "is_hybrid", False))
    spin = bool(getattr(scf, "spin_polarized", False))
    if _has_hubbard(problem):
        found = _refused(problem, LDAU_SWITCH, LDAU_REASON, features=features, refused=hybrid or spin)
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


def ts_ignore_symmetry_note(translation, *, backend_is_cupy) -> str | None:
    """The NOTE of a TS cell whose ``Ignore_Symmetry`` does not keep it off the sectors, or ``None``.

    PARSEC 1.5 refuses the TS correction unless symmetry is switched off (``Ignore_Symmetry: true``, error 159:
    its Hirshfeld sums read a density it would otherwise hold on the irreducible wedge), so every TS input says
    so.  Here the correction reads the density on the full grid of the cell whatever the route of the loop, and the
    sectors give the energy of the full grid; under ``PARSEC_GAMMA_TS_SECTORS`` ``auto`` or ``on`` (the default is
    ``off``) the command line therefore reads that keyword of a TS input as PARSEC's requirement, not as a request
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
    "LDAU_SWITCH",
    "LIBXC_SWITCH",
    "META_GGA_SWITCH",
    "TS_SWITCH",
    "correction_refusal",
    "gamma_feature_setting",
    "gamma_feature_settings",
    "meta_gga_or_libxc_refusal",
    "taken_features",
    "ts_ignore_symmetry_note",
    "MINIMUM_DEVICES",
    "MINIMUM_ENTRIES",
    "basis_entries",
    "process_device_count",
]
