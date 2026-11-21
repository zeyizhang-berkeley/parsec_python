"""Every translation of a perfect supercell: the mesh and the wedge of ``PARSEC_SUPERCELL_KPOINTS=auto``.

A Gamma-point cell that repeats a smaller orthorhombic cell ``m_1 x m_2 x m_3`` times has ``m_1 m_2 m_3`` pure
translations.  The Gamma route keeps those by half the cell, whose characters are real
(``parsec_python.acceleration.Symmetry.axis_reflection``): along an axis of odd repeats none.  Bloch's theorem keeps
them all: with the Bloch block of the finite-difference Laplacian as the kinetic term of a k-point
(``PARSEC_KPOINT_KINETIC=bloch``) the Gamma point of the supercell is the cell of ``m_a / K_a`` repeats with the
Gamma-centred mesh ``K_1 x K_2 x K_3``, for every divisor ``K_a`` of ``m_a``: the same eigenvalues, the total energy
``1/K`` of the supercell's (``K = K_1 K_2 K_3``).  Each k-point is a complex Hermitian problem of ``1/K`` of the
supercell's rows and states; the irreducible ones are about ``K/48`` for a cubic crystal.  This module finds the
repeats (from the counts of the supercell's grid, which it does not build) and builds the cell of a mesh, with
``1/K`` of ``Net_Charges`` and of the states (:func:`parsec_python.SCF.kpoint_supercell.kpoints_of_supercell` is the
smallest cell with the mesh of all the repeats); a cell whose grid the builder refuses (too small for the stencil of
``Expansion_Order``) is not taken.  Two switches shape that run (both read only under
``PARSEC_SUPERCELL_KPOINTS=auto``, both checked in every run):

* ``PARSEC_SUPERCELL_KPOINTS_MESH``: ``all`` (default) the mesh of all the repeats on the smallest cell, as before
  2026-11-19; ``K1xK2xK3`` a mesh whose entries divide the repeats (a larger cell of fewer k-points, each a larger
  problem; an error where that cell cannot take the grid); ``auto`` the such mesh of most points with an entry above
  two whose cell holds at least ``2^22`` entries of grid points times states (k-points below that size cost a device
  about the same whatever their size), else the mesh of all the repeats where it has an entry above two, else the
  smallest larger cell that takes the grid, else the Gamma point of the supercell.  Where the cell of ``all`` cannot
  take the grid, the supercell runs at the Gamma point with a note.
* ``PARSEC_SUPERCELL_KPOINTS_WEDGE``: ``auto`` (default) reduces the mesh of a single point to the irreducible wedge of
  the operations of the cell that map its grid and the mesh onto themselves, with time reversal, and symmetrizes the
  density with them, as an input of that cell and mesh is read (``Input.parsec_input._default_routines_wedge``):
  diamond's cube with ``n x n x n`` points, ``(n+1)(n+3)(n+5)/48`` k-points for an odd ``n`` instead of
  ``(n^3+1)/2``; ``off`` time reversal alone, as before 2026-11-19.  Forces, relaxations and molecular dynamics,
  ``Ignore_Symmetry`` and a cell without operations keep time reversal alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os

import numpy as np

MESH_SWITCH = "PARSEC_SUPERCELL_KPOINTS_MESH"
WEDGE_SWITCH = "PARSEC_SUPERCELL_KPOINTS_WEDGE"
# grid points x states of a cell from which PARSEC_SUPERCELL_KPOINTS_MESH=auto takes it: below that a k-point costs a
# device about the same whatever its size (A100, 2026-11-19: diamond k-points of 1.6e5 entries 0.27 s per SCF, of
# 6.3e6 0.88 s, of 6.9e7 2.0-2.8 s, of 3.8e8 5.6 s, of 1.5e9 13.4 s; one SCF of 8 iterations on 4 GPUs)
AUTO_MESH_ENTRIES = 2 ** 22


def _parse_mesh(value: str) -> tuple[int, int, int] | None:
    parts = value.replace("*", "x").split("x")
    if len(parts) != 3:
        return None
    try:
        mesh = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if any(entry < 1 for entry in mesh):
        return None
    return mesh  # type: ignore[return-value]


def supercell_mesh_setting() -> str | tuple[int, int, int]:
    """``PARSEC_SUPERCELL_KPOINTS_MESH``: ``"all"`` (default), ``"auto"`` or a mesh ``(K1, K2, K3)``."""
    value = os.environ.get(MESH_SWITCH, "all").strip().lower() or "all"
    if value in {"all", "auto"}:
        return value
    mesh = _parse_mesh(value)
    if mesh is None or int(np.prod(mesh)) == 1:
        raise ValueError(
            f"{MESH_SWITCH} must be all, auto or a mesh K1xK2xK3 of positive whole numbers, not all 1 "
            f"(e.g. 3x3x3), not {value!r}"
        )
    return mesh


def supercell_wedge_setting() -> str:
    """``PARSEC_SUPERCELL_KPOINTS_WEDGE``: ``auto`` (default) the irreducible wedge of the cell, ``off`` time reversal."""
    value = os.environ.get(WEDGE_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{WEDGE_SWITCH} must be auto or off, not {value!r}")
    return value


def validate_settings() -> None:
    """Check the values of ``PARSEC_SUPERCELL_KPOINTS`` and of the two switches of this module (an error before
    anything is prepared)."""
    from .kpoint_supercell import supercell_kpoints_setting

    supercell_kpoints_setting()
    supercell_mesh_setting()
    supercell_wedge_setting()


@dataclass(frozen=True)
class SupercellMesh:
    """The cell and mesh that solve a repeating Gamma-point cell (:func:`supercell_mesh_route`)."""

    problem: object
    """The cell with the Gamma-centred mesh ``mesh`` (its ``space_group`` the wedge's, or ``None``)."""
    mesh: tuple[int, int, int]
    repeats: tuple[int, int, int]
    """The repeats of the smallest cell along each axis (``mesh`` divides them)."""
    wedge: int | None
    """The order of the group of the wedge, ``None`` for time reversal alone."""
    notes: tuple[str, ...] = ()

    @property
    def cells(self) -> int:
        return int(np.prod(self.mesh))

    def describe(self) -> str:
        """The note of the output: what was solved and the factor between the two total energies."""
        m, r = self.mesh, self.repeats
        inner = tuple(int(a // b) for a, b in zip(r, m))
        cell = "that cell" if inner == (1, 1, 1) else f"the cell of {inner[0]} x {inner[1]} x {inner[2]} of them"
        if self.wedge is None:
            reduction = "the mesh reduced by time reversal alone"
        else:
            reduction = (
                f"the mesh reduced to the irreducible wedge of {self.wedge} operations of the cell and time reversal, "
                "the density symmetrized with them"
            )
        charge = float(getattr(self.problem.scf, "net_charge", 0.0) or 0.0)
        charged = f"; Net_Charges {charge:.10g} in the cell, 1/{self.cells} of the supercell's" if charge else ""
        return (
            f"the cell repeats a smaller one {r[0]} x {r[1]} x {r[2]} times: it is solved as {cell} with the "
            f"Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} (PARSEC_SUPERCELL_KPOINTS=auto, "
            f"{MESH_SWITCH}; the Bloch kinetic term; {reduction}{charged}): the total energy reported is that of the "
            f"cell solved, 1/{self.cells} of the supercell's"
        )


def _supercell_counts(problem) -> np.ndarray | None:
    """The points of the supercell's grid along its axes, without building that grid; ``None`` where the rule of the
    points refuses the cell (its run then reports that itself).

    On a bulk orthorhombic cell of the default routines (what
    :func:`~parsec_python.SCF.kpoint_supercell._bloch_kinetic_applies` admits) ``Grid.pbc.default_periodic_grid_counts``
    gives the counts of ``Grid.pbc.build_periodic_grid``; building the grid of the supercell to read them (as before
    2026-11-19) took 0.5 s for diamond 12^3 and grows with its points.
    """
    from ..Grid.pbc import default_periodic_grid_counts

    try:
        return np.asarray(default_periodic_grid_counts(problem.periodic_cell, problem.grid), dtype=np.int64)
    except ValueError:
        return None


def _supercell_repeats(problem, tolerance: float = 1.0e-7):
    """``(repeats, counts)`` of a Gamma-point cell that repeats a smaller orthorhombic one, else ``None``.

    The repeats ``m_a`` are the largest whole divisions of each axis (of its grid points and of every species' atom
    count) whose translation maps the atoms, species by species, modulo the cell; ``counts`` are the grid points of
    the supercell.  ``None`` under ``PARSEC_SUPERCELL_KPOINTS=off``, with k-points already, without a repeat, under an
    explicit ``PARSEC_KPOINT_KINETIC=gradient`` (the caller runs the mesh with ``bloch``, which reproduces the
    supercell), and for a cell or calculation that the Bloch kinetic term does not reproduce (an oblique cell, a slab,
    the Fortran-derived routines, spin, spin-orbit, hybrid or meta-GGA functionals, fixed occupations).
    """
    from .kpoint_supercell import _bloch_kinetic_applies, _plain_scalar, _repeat_of_axis, supercell_kpoints_setting

    if supercell_kpoints_setting() != "auto" or problem.periodic_cell is None:
        return None
    if getattr(problem, "has_kpoint_sampling", False):
        return None
    # the mesh reproduces the supercell with the Bloch kinetic term only (the caller sets it for the run)
    if (os.environ.get("PARSEC_KPOINT_KINETIC", "").strip().lower() or "unset") == "gradient":
        return None
    if not _bloch_kinetic_applies(problem) or not _plain_scalar(problem):
        return None
    counts = _supercell_counts(problem)
    if counts is None:
        return None
    sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    by_species: dict = {}
    for atom in problem.atoms:
        by_species.setdefault(atom.symbol, []).append(np.asarray(atom.position, dtype=np.float64))
    by_species = {name: np.asarray(points) for name, points in by_species.items()}
    by_species["__period__"] = sides
    repeats = tuple(
        int(_repeat_of_axis(by_species, float(sides[axis]), axis, int(counts[axis]), tolerance)) for axis in range(3)
    )
    if int(np.prod(repeats)) == 1:
        return None
    return repeats, counts


def _cell_of_mesh(problem, mesh, counts, errors: list[str] | None = None) -> object | None:
    """The cell of ``problem`` (a Gamma-point supercell of ``counts`` grid points) whose Gamma-centred mesh ``mesh``
    (a divisor of its repeats) reproduces it, else ``None`` (the reason appended to ``errors``).

    The cell keeps the atoms in it, ``1/K`` of ``Net_Charges`` (``K`` the points of the mesh: with the uniform
    background of a charged cell the same density and energy per cell), ``1/K`` of the states rounded up and six
    more for the filters (unoccupied: the energies are those of the supercell; no count where the supercell has none,
    so that its run takes the default of the cell), and the steps of the supercell's grid: the spacing of the
    supercell, or a hair below it where round-off gives the cell fewer points under PARSEC's rule of the points
    (``Grid.pbc.fortran_points_per_axis``: 16 for the cube of diamond of 18 within 3, 5 or 6 cubes).  A spacing whose
    grid the builder refuses (a rounded-down count at the limit of the stencil of ``Expansion_Order``) is passed over
    like one of another count; ``None`` where neither gives the cell the supercell's grid, among others a cell too
    small for the stencil.
    """
    from .kpoint_supercell import _grid_counts

    mesh = tuple(int(value) for value in mesh)
    count = int(np.prod(mesh))
    counts = np.asarray(counts, dtype=np.int64)
    sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    cell_sides = sides / np.asarray(mesh, dtype=np.float64)
    atoms = [
        atom for atom in problem.atoms
        if np.all(np.mod(np.asarray(atom.position, dtype=np.float64), sides) < cell_sides - 1.0e-9 * sides)
    ]
    if len(atoms) * count != len(problem.atoms):
        if errors is not None:
            errors.append(f"the cell of the mesh {'x'.join(map(str, mesh))} holds {len(atoms)} of the "
                          f"{len(problem.atoms)} atoms, not 1/{count}")
        return None
    requested = problem.scf.number_of_states
    scf_changes: dict = {"number_of_states": None if requested is None else int(math.ceil(int(requested) / count)) + 6}
    charge = float(getattr(problem.scf, "net_charge", 0.0) or 0.0)
    if charge != 0.0:
        # Net_Charges is that of the supercell: 1/K of it in each cell
        scf_changes["net_charge"] = charge / count
    step = float(np.min(sides / counts))
    for spacing in (step, step * (1.0 - 1.0e-9)):
        cell = replace(
            problem,
            atoms=[type(atom)(atom.symbol, tuple(np.mod(np.asarray(atom.position, float), sides))) for atom in atoms],
            periodic_cell=replace(problem.periodic_cell, lattice_vectors=np.diag(cell_sides)),
            grid=replace(problem.grid, spacing=spacing),
            scf=replace(problem.scf, **scf_changes),
            # the space group of the supercell is not the cell's (time reversal alone; the wedge of the cell below)
            space_group=None,
            monkhorst_pack_dimensions=mesh,
            monkhorst_pack_shift=(0.0, 0.0, 0.0),
        )
        try:
            built = _grid_counts(cell)
        except ValueError as error:
            if errors is not None:
                errors.append(str(error))
            continue
        if np.array_equal(built * np.asarray(mesh), counts):
            return cell
    if errors is not None and not errors:
        errors.append(f"no spacing gives the cell of the mesh {'x'.join(map(str, mesh))} "
                      f"{(counts // np.asarray(mesh)).tolist()} points")
    return None


def smallest_cell(problem, tolerance: float = 1.0e-7):
    """``(cell problem, repeats)``: the smallest repeating cell of a Gamma-point supercell with the Gamma-centred mesh
    of all its repeats (:func:`_supercell_repeats`, :func:`_cell_of_mesh`), else ``None``; the implementation of
    :func:`parsec_python.SCF.kpoint_supercell.kpoints_of_supercell`."""
    found = _supercell_repeats(problem, tolerance)
    if found is None:
        return None
    repeats, counts = found
    cell = _cell_of_mesh(problem, repeats, counts)
    return None if cell is None else (cell, repeats)


def _divisors(value: int) -> list[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def _supercell_states(problem) -> int:
    """The states of the supercell: ``States_Num``, else the default of the SCF, ``ceil(N_e/2) + 6``
    (``parsec_python.cli.planned_state_count``) from the valence charges of the pseudopotentials."""
    requested = problem.scf.number_of_states
    if requested is not None:
        return int(requested)
    import warnings

    from ..V_ion.ionic_potential import ionic_charge, load_pseudopotentials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = load_pseudopotentials(
            problem.pseudopotentials, xc_functional=problem.scf.xc_functional, check_correlation=False
        )
    electrons = ionic_charge(tuple(problem.atoms), tables) - float(getattr(problem.scf, "net_charge", 0.0) or 0.0)
    return int(math.ceil(0.5 * electrons)) + 6


def _auto_meshes(problem, repeats, counts) -> list[tuple[int, int, int]]:
    """The meshes of ``PARSEC_SUPERCELL_KPOINTS_MESH=auto`` in the order they are tried (the first whose cell takes the
    grid of the input is used; none: the Gamma point of the supercell).

    The meshes that divide ``repeats``, the same entry along axes of the same side and repeats (the cell keeps the
    shape of the smallest one, so its point group), not of ones and twos alone (the time-reversal invariant k-points,
    which the Gamma route takes as the translations by half the cell, in real arithmetic): first those whose cell
    holds at least :data:`AUTO_MESH_ENTRIES` grid points times states (k-points large enough to use a device), the
    mesh of most points first; then the mesh of all the repeats (k-points that small cost about the same whatever
    their size, so the fewest), unless it is of ones and twos alone; then the other meshes, of most points first (a
    smallest cell too small for the stencil of ``Expansion_Order``: the smallest larger one that holds it).
    """
    total_points = int(np.prod(counts))
    states = _supercell_states(problem)
    sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    repeats = tuple(int(value) for value in repeats)
    large: list = []
    small: list = []
    for mesh in (
        (a, b, c) for a in _divisors(repeats[0]) for b in _divisors(repeats[1]) for c in _divisors(repeats[2])
    ):
        if all(entry <= 2 for entry in mesh):
            continue
        if mesh == repeats:
            continue  # after the large cells
        if any(
            mesh[i] != mesh[j]
            for i in range(3) for j in range(i + 1, 3)
            if repeats[i] == repeats[j] and abs(sides[i] - sides[j]) <= 1.0e-9 * max(sides[i], sides[j])
        ):
            continue
        cells = int(np.prod(mesh))
        entries = (total_points // cells) * (int(math.ceil(states / cells)) + 6)
        (large if entries >= AUTO_MESH_ENTRIES else small).append((cells, mesh))
    order = [mesh for _, mesh in sorted(large, reverse=True)]
    if not all(entry <= 2 for entry in repeats):
        order.append(repeats)
    order += [mesh for _, mesh in sorted(small, reverse=True)]
    return order


def _auto_mesh(problem, repeats, counts) -> tuple[int, int, int] | None:
    """The first mesh :func:`_auto_meshes` tries, ``None`` for the Gamma point of the supercell."""
    order = _auto_meshes(problem, repeats, counts)
    return order[0] if order else None


def supercell_mesh_route(translation, notes: list[str] | None = None) -> SupercellMesh | None:
    """The cell and mesh of ``PARSEC_SUPERCELL_KPOINTS=auto`` for this input, ``None`` where that route does not
    apply (:func:`_supercell_repeats`; the supercell's grid is not built).

    ``notes``: where the cell repeats a smaller one but no cell of the mesh takes the grid of the input (the mesh of
    all the repeats on a cell too small for the stencil of ``Expansion_Order``), or ``PARSEC_SUPERCELL_KPOINTS_MESH=auto``
    leaves the supercell to the Gamma route, the reason is appended to it and the route returns ``None``: the Gamma
    point of the supercell, as without the switch.

    Raises ``ValueError`` for a value of a switch that is not known and for a mesh of
    ``PARSEC_SUPERCELL_KPOINTS_MESH`` that does not divide the repeats of the cell or whose cell cannot take the grid.
    """

    validate_settings()
    problem = translation.problem
    found = _supercell_repeats(problem)
    if found is None:
        return None
    repeats, counts = found
    repeats = tuple(int(value) for value in repeats)
    repeated = f"the cell repeats a smaller one {repeats[0]} x {repeats[1]} x {repeats[2]} times"
    setting = supercell_mesh_setting()
    errors: list[str] = []
    cell = None
    if setting == "all":
        mesh = repeats
        cell = _cell_of_mesh(problem, mesh, counts, errors)
        if cell is None:
            if notes is not None:
                notes.append(
                    f"PARSEC_SUPERCELL_KPOINTS=auto: {repeated}, but that cell cannot take the grid of the input "
                    f"({errors[-1]}): the Gamma point of the supercell ({MESH_SWITCH}=auto, or a mesh K1xK2xK3 of "
                    "fewer points, takes a larger cell)"
                )
            return None
    elif setting == "auto":
        for mesh in _auto_meshes(problem, repeats, counts):
            cell = _cell_of_mesh(problem, mesh, counts, errors)
            if cell is not None:
                break
        if cell is None:
            if notes is not None:
                reason = (f"no cell of a mesh above two takes the grid of the input ({errors[-1]})" if errors
                          else "its repeats are ones and twos, whose translations by half the cell the Gamma route "
                          "takes in real arithmetic")
                notes.append(f"PARSEC_SUPERCELL_KPOINTS=auto, {MESH_SWITCH}=auto: {repeated}, but {reason}: the "
                             "Gamma point of the supercell")
            return None
    else:
        mesh = setting
        if any(r % k for r, k in zip(repeats, mesh)):
            raise ValueError(
                f"{MESH_SWITCH}={'x'.join(map(str, mesh))}: the cell repeats a smaller one "
                f"{repeats[0]} x {repeats[1]} x {repeats[2]} times, and the mesh must divide these repeats"
            )
        cell = _cell_of_mesh(problem, mesh, counts, errors)
        if cell is None:
            raise ValueError(
                f"{MESH_SWITCH}={'x'.join(map(str, mesh))}: {repeated}, but the cell of that mesh cannot take the "
                f"grid of the input ({errors[-1]}); a mesh of fewer points takes a larger cell"
            )
    cell_notes: list[str] = []
    wedge = None
    moving = bool(
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    )
    if supercell_wedge_setting() == "auto" and not moving and not getattr(translation, "ignore_symmetry", False):
        from ..Input.parsec_input import ParsecInputError, _default_routines_wedge

        try:
            symmetrize, group = _default_routines_wedge(
                cell.atoms,
                cell.periodic_cell,
                cell.grid,
                (tuple(mesh), (0.0, 0.0, 0.0)),
                ignore_symmetry=False,
                symmetrize_density=True,
                symmetrize_given=False,
                warnings=cell_notes,
            )
        except (ParsecInputError, ValueError) as error:
            cell_notes.append(f"{WEDGE_SWITCH}: no operations of the cell ({error}); the mesh is reduced by time "
                              "reversal")
            group = None
        if group is not None:
            cell = replace(cell, space_group=group, scf=replace(cell.scf, symmetrize_density=bool(symmetrize)))
            wedge = int(group.order)
        # the reader's note names the block of an input of that cell; this one has none
        prefix = "Monkhorst_Pack_Shift: "
        cell_notes = [f"{WEDGE_SWITCH}: {note[len(prefix):]}" if note.startswith(prefix) else note
                      for note in cell_notes]
    return SupercellMesh(problem=cell, mesh=tuple(int(v) for v in mesh), repeats=repeats, wedge=wedge,
                         notes=tuple(cell_notes))


__all__ = [
    "AUTO_MESH_ENTRIES",
    "MESH_SWITCH",
    "SupercellMesh",
    "WEDGE_SWITCH",
    "smallest_cell",
    "supercell_mesh_route",
    "supercell_mesh_setting",
    "supercell_wedge_setting",
    "validate_settings",
]
