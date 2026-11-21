"""Opt-in symmetrization of the input positions (``PARSEC_SYMMETRIZE_POSITIONS``).

A structure given with rounded coordinates is symmetric only to the rounding: the symmetry detection takes the
operations that carry every atom within ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` onto an atom of its species, but the
forces and the moves of a calculation in sectors check the atoms to ``1e-6`` bohr
(:data:`parsec_python.Forces.symmetry.POSITION_TOLERANCE`), and the sectors solve the Hamiltonian averaged over the
group rather than that of the input.  Under ``PARSEC_SYMMETRIZE_POSITIONS=on`` the positions are replaced, before
anything is prepared, by their average over the group of operations that map the atoms within that tolerance:

``R'_a = (1/|G|) sum_g g R_{g^-1 a}``,

which every operation of ``G`` leaves invariant (to the round-off of the coordinates).  The structure then has the
group exactly, and the detection, the sectors, the forces and a relaxation all take it.

* A cluster: the signed permutations of the axes (the 48 operations of the cube,
  :func:`parsec_python.Forces.symmetry.atom_symmetry_operations`) about the origin of the calculation, the centre of
  the bounding box where the input is recentred (``center_cluster_geometry``, which the preparation applies again:
  the symmetric structure keeps its group about the new centre, which lies on every axis of the group).
* A bulk cell: the operations ``r -> R r + t`` with ``R`` an orthogonal map of the lattice onto itself and ``t`` any
  translation, the atoms matched modulo the lattice (the space group of the atoms, with the pure translations of a
  supercell); its group is the union of cosets ``T (R, t_R)`` of the translations ``T``, and the average is taken
  over ``T`` and then over the coset representatives.  Wires and slabs keep their positions (a note says so).

  The average over ``T`` keeps the image tables of the generators of ``T`` only (a few rows of ``N`` atoms): the
  orbits of ``T`` and the translation that carries the first atom of an orbit onto each of its atoms follow from them,
  and every atom goes to the mean of its orbit carried by that translation, in ``O(N)`` memory.  The translation of
  an operation is the one that carries the atoms onto their images best (its mean residual), so that the average is
  invariant under every operation exactly; such an average floats as a whole by the mean of the noise of the input.
  It is then shifted as a whole (which keeps it invariant) so that each operation whose translation lies within
  ``4 x`` the tolerance of one that maps the grid of the calculation onto itself has that translation exactly: the
  operations about the origin, about a site of the grid or with a translation of whole grid steps, which the
  detection of the sectors takes (shift ``s`` with ``(I - R) s`` the offsets, least squares, the operations taken in
  their order while they agree and ``|s|`` stays within the tolerance).

The group is found within ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` (1e-7 bohr by default: the positions then move by at
most that).  Operations found within a loose tolerance need not form a group; the average is then not invariant under
all of them, and the run is refused before the SCF with that reason.  ``off`` (the default) keeps the positions of the
input, as PARSEC 1.5 does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .symmetry import AtomSymmetryOperation, atom_symmetry_operations, symmetrize_forces

SYMMETRIZE_POSITIONS_SWITCH = "PARSEC_SYMMETRIZE_POSITIONS"

#: Largest change of a symmetrized position under an operation of its group, relative to the size of the structure
#: (bohr per bohr): the average of a group is invariant to the round-off of the coordinates.
_INVARIANCE_TOLERANCE = 1.0e-10

#: An operation of a cell whose translation lies within this many times the tolerance of a translation that maps the
#: grid onto itself gets that translation exactly (the structure shifted as a whole).
_GRID_PIN_FACTOR = 4.0

#: Candidate translations of a point operation of a cell tried at once (the first that maps every atom represents it).
_CANDIDATE_CHUNK = 4096


def symmetrize_positions_setting() -> str:
    """``PARSEC_SYMMETRIZE_POSITIONS``: ``off`` (the default) or ``on``; see the module."""

    value = os.environ.get(SYMMETRIZE_POSITIONS_SWITCH, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{SYMMETRIZE_POSITIONS_SWITCH} must be off or on, not {value!r}")
    return value


@dataclass(frozen=True)
class PositionSymmetrization:
    """What :func:`symmetrize_input_positions` did: the order of the group, the tolerance and the largest move."""

    order: int
    tolerance: float
    largest_move: float
    kind: str
    """``cluster`` or ``cell``."""
    pinned: int = 0
    """A cell: the operations (besides the pure translations) whose translation the shift onto the grid made exact."""
    shift: float = 0.0
    """A cell: the length (bohr) of that shift."""

    def note(self) -> str:
        if self.order <= 1:
            return (
                f"NOTE: {SYMMETRIZE_POSITIONS_SWITCH}=on: only the identity maps the atoms within "
                f"{self.tolerance:g} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE); the positions are kept"
            )
        where = "about the origin of the calculation" if self.kind == "cluster" else "of the cell (modulo its lattice)"
        text = (
            f"NOTE: {SYMMETRIZE_POSITIONS_SWITCH}=on: the positions are averaged over the {self.order} operations "
            f"{where} that map the atoms within {self.tolerance:g} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE)"
        )
        if self.kind == "cell":
            text += (
                f" and shifted as a whole by {self.shift:.3e} bohr, which gives {self.pinned} point operations a "
                "translation that maps the grid onto itself"
            )
        return text + f"; the largest move is {self.largest_move:.3e} bohr"


def _invariance_error(positions: np.ndarray, operations: Sequence[AtomSymmetryOperation]) -> float:
    """The largest distance (bohr) from an image of an atom under an operation of a cluster to its image atom."""

    largest = 0.0
    for operation in operations:
        moved = positions @ np.asarray(operation.matrix, dtype=np.float64).T
        largest = max(largest, float(np.max(np.abs(moved - positions[operation.image]), initial=0.0)))
    return largest


def symmetrized_cluster_positions(positions: np.ndarray, symbols: Sequence[str], tolerance: float):
    """``(positions, order)``: the average of ``positions`` over the signed permutations about the origin that map
    every atom within ``tolerance`` onto an atom of its species.  Raises ``ValueError`` where those operations form no
    group (the average is not invariant under them)."""

    from ..models import Atom

    positions = np.asarray(positions, dtype=np.float64)
    atoms = [Atom(symbol, position) for symbol, position in zip(symbols, positions)]
    operations = atom_symmetry_operations(atoms, tolerance=tolerance)
    if len(operations) <= 1:
        return positions.copy(), 1
    symmetric = symmetrize_forces(positions, operations)
    scale = max(1.0, float(np.max(np.abs(symmetric), initial=0.0)))
    largest_move = float(np.max(np.linalg.norm(symmetric - positions, axis=1), initial=0.0))
    if _invariance_error(symmetric, operations) > _INVARIANCE_TOLERANCE * scale or largest_move > 2.0 * tolerance:
        raise ValueError(
            f"{SYMMETRIZE_POSITIONS_SWITCH}=on: the {len(operations)} operations that map the atoms within "
            f"{tolerance:g} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE) form no group within it, so their average is not a "
            "symmetric structure near the input; lower the tolerance"
        )
    return symmetric, len(operations)


def _cell_matches(fractions, symbols, trees, members, vectors, image_fractions, tolerance, rows=None):
    """The image table of ``image_fractions`` (fractional, one row per atom or per atom of ``rows``) onto the atoms of
    their species within ``tolerance`` bohr modulo the lattice, or ``None``."""

    rows = np.arange(len(symbols)) if rows is None else np.asarray(rows)
    image = np.full(rows.size, -1, dtype=np.int64)
    for symbol, indices in members.items():
        local = np.flatnonzero(np.isin(rows, indices))
        if not local.size:
            continue
        queries = np.mod(image_fractions[local], 1.0)
        queries[queries >= 1.0] = 0.0
        _, nearest = trees[symbol].query(queries, k=1)
        found = indices[nearest]
        difference = image_fractions[local] - fractions[found]
        difference -= np.round(difference)
        if np.any(np.linalg.norm(difference @ vectors, axis=1) > tolerance):
            return None
        image[local] = found
    return image


def _mean_translation(positions, vectors, inverse, matrix, translation, image):
    """The translation ``t`` of ``r -> R r + t`` that carries the atoms onto their images best: ``translation`` plus
    the mean of the residuals ``R_image - R r - translation`` (each by the nearest lattice vector).  These means
    compose as the operations do (``t_gh = R_g t_h + t_g``), so that the average over a group with them is invariant
    under it exactly, whatever the noise of the positions the operations were found on.  A pure translation of order
    ``n`` gets ``n t`` a lattice vector (the residuals of a cycle add up to one), a fraction of the lattice exactly."""

    residual = positions[image] - positions @ matrix.T - translation
    residual -= np.round(residual @ inverse) @ vectors
    return translation + residual.mean(axis=0)


def _cell_average(positions, vectors, operations):
    """``(1/|S|) sum_s s R`` over ``operations`` ``(matrix, translation, image)`` (``r -> R r + t``, ``t`` the mean
    translation of :func:`_mean_translation` for these positions), the lattice vector between an image and its atom
    removed."""

    total = np.zeros_like(positions)
    inverse = np.linalg.inv(vectors)
    for matrix, translation, image in operations:
        moved = positions @ matrix.T + _mean_translation(positions, vectors, inverse, matrix, translation, image)
        steps = np.round((moved - positions[image]) @ inverse)
        total[image] += moved - steps @ vectors
    return total / float(len(operations))


def _generator_translations(positions, vectors, generators, offsets):
    """``(means, mismatch)``: the mean translations (Cartesian, :func:`_mean_translation`) of the pure translations
    with the image tables ``generators`` and approximate ``offsets``, and the largest distance (bohr) of an image of
    an atom from its image atom under one of them."""

    inverse = np.linalg.inv(vectors)
    identity = np.eye(3)
    means = np.zeros((len(generators), 3))
    mismatch = 0.0
    for index, (table, offset) in enumerate(zip(generators, offsets)):
        means[index] = _mean_translation(positions, vectors, inverse, identity, offset, table)
        residual = positions[table] - positions - means[index]
        residual -= np.round(residual @ inverse) @ vectors
        mismatch = max(mismatch, float(np.max(np.linalg.norm(residual, axis=1), initial=0.0)))
    return means, mismatch


def _translation_orbits(vectors, generators, means):
    """``(representative, translation)``: for every atom the first atom of its orbit under the pure translations
    generated by the image tables ``generators``, and the translation (Cartesian, folded into the cell) that carries
    that atom onto it, the ``means`` of the generators added along a path of the orbit.  Labels spread along the
    generators until none changes; memory ``O(N)``."""

    inverse = np.linalg.inv(vectors)
    count = len(generators[0])
    representative = np.arange(count, dtype=np.int64)
    translation = np.zeros((count, 3))
    changed = True
    while changed:
        changed = False
        for table, mean in zip(generators, means):
            # forward: the image g(a) of an atom takes the label of a where it is smaller (g is a permutation)
            better = representative < representative[table]
            if np.any(better):
                targets = table[better]
                representative[targets] = representative[better]
                translation[targets] = translation[better] + mean
                changed = True
            # backward: an atom takes the label of its image where that is smaller
            image_labels = representative[table]
            better = image_labels < representative
            if np.any(better):
                image_translations = translation[table]
                representative[better] = image_labels[better]
                translation[better] = image_translations[better] - mean
                changed = True
        translation -= np.floor(translation @ inverse) @ vectors
    return representative, translation


def _translation_average(positions, vectors, representative, translation):
    """The average of ``positions`` over the pure translations of the orbits ``(representative, translation)`` of
    :func:`_translation_orbits`: every atom at the mean over its orbit of ``x_j - t_j``, plus its own ``t_i``."""

    inverse = np.linalg.inv(vectors)
    count = len(positions)
    carried = positions - translation
    # x_j - t_j lies near x_rep modulo the lattice: the lattice vector between them removed
    carried -= np.round((carried - positions[representative]) @ inverse) @ vectors
    sizes = np.bincount(representative, minlength=count).astype(np.float64)
    centre = np.zeros_like(positions)
    occupied = sizes > 0
    for axis in range(3):
        sums = np.bincount(representative, weights=carried[:, axis], minlength=count)
        centre[occupied, axis] = sums[occupied] / sizes[occupied]
    averaged = centre[representative] + translation
    averaged -= np.round((averaged - positions) @ inverse) @ vectors
    return averaged


def _cell_invariance_error(positions, vectors, operations) -> float:
    """The largest distance (bohr) from an image of an atom to its image atom, modulo the lattice, under the
    operations with their mean translations for ``positions``."""

    inverse = np.linalg.inv(vectors)
    largest = 0.0
    for matrix, translation, image in operations:
        moved = positions @ matrix.T + _mean_translation(positions, vectors, inverse, matrix, translation, image)
        difference = (moved - positions[image]) @ inverse
        difference -= np.round(difference)
        largest = max(largest, float(np.max(np.abs(difference @ vectors), initial=0.0)))
    return largest


def _grid_shift(positions, vectors, representatives, fractional_matrices, pure_translations, grid, tolerance):
    """``(s, pinned)``: the shift to subtract from the positions of a cell invariant under ``representatives`` (with
    their mean translations) so that every one whose coset of translations comes within ``_GRID_PIN_FACTOR x
    tolerance`` of a translation that maps the grid ``(counts, shift)`` onto itself has that translation exactly,
    and the number of them.

    Shifting the positions by ``-s`` changes the translation of ``r -> R r + t`` into ``t - (I - R) s`` and leaves the
    structure invariant.  A grid point has the fractional coordinates ``(n + shift)/counts``; ``f -> f N + t`` maps the
    grid onto itself when ``diag(1/counts) N diag(counts)`` is an integer matrix and ``t`` lies in
    ``shift/counts - (shift/counts) N + Z^3/counts``.  The operations are taken in their order while the equations
    ``(I - R) s = offset`` agree (least squares of the smallest ``s``, residual within the round-off) and ``s`` stays
    within ``tolerance`` (the input floats by the mean of its noise, less than that)."""

    counts = np.asarray(grid[0], dtype=np.float64).reshape(3)
    grid_shift = np.asarray(grid[1], dtype=np.float64).reshape(3)
    inverse = np.linalg.inv(vectors)
    scale = max(1.0, float(np.max(np.abs(vectors))))
    pure = np.asarray(pure_translations, dtype=np.float64).reshape(-1, 3) @ inverse
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    solution = np.zeros(3)
    pinned = 0
    for (matrix, translation, image), fractional in zip(representatives, fractional_matrices):
        fractional = np.asarray(fractional, dtype=np.float64)
        if np.array_equal(fractional, np.eye(3)):
            continue
        scaled = fractional / counts[:, None] * counts[None, :]
        if np.any(np.abs(scaled - np.round(scaled)) > 1.0e-9):
            continue  # R does not map the grid onto itself
        mean = _mean_translation(positions, vectors, inverse, matrix, translation, image)
        base = grid_shift / counts - (grid_shift / counts) @ fractional
        steps = (mean @ inverse + pure - base) * counts
        offsets = (steps - np.round(steps)) / counts
        lengths = np.linalg.norm(offsets @ vectors, axis=1)
        best = int(np.argmin(lengths))
        if lengths[best] > _GRID_PIN_FACTOR * tolerance:
            continue
        trial_rows = rows + [np.eye(3) - matrix]
        trial_values = values + [offsets[best] @ vectors]
        system = np.vstack(trial_rows)
        target = np.concatenate(trial_values)
        trial = np.linalg.lstsq(system, target, rcond=None)[0]
        if (
            float(np.max(np.abs(system @ trial - target))) > _INVARIANCE_TOLERANCE * scale
            or float(np.linalg.norm(trial)) > tolerance
        ):
            continue
        rows, values, solution = trial_rows, trial_values, trial
        pinned += 1
    return solution, pinned


def _symmetrized_cell(positions, symbols: Sequence[str], lattice_vectors, tolerance: float, grid=None):
    """``(positions, order, pinned, shift)`` of :func:`symmetrized_cell_positions`, with the operations whose
    translation the shift onto ``grid`` made exact and the length of that shift (none without a grid)."""

    from .periodic import _lattice_point_operations

    positions = np.asarray(positions, dtype=np.float64)
    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(vectors)
    fractions = positions @ inverse
    symbols = list(symbols)
    members = {symbol: np.flatnonzero([item == symbol for item in symbols]) for symbol in dict.fromkeys(symbols)}
    wrapped = np.mod(fractions, 1.0)
    wrapped[wrapped >= 1.0] = 0.0
    trees = {symbol: cKDTree(wrapped[indices], boxsize=1.0) for symbol, indices in members.items()}
    # the rarest species gives the candidate translations: the image of its first atom is one of its atoms
    rare = min(members, key=lambda symbol: members[symbol].size)
    anchor = int(members[rare][0])
    # a few atoms of every species, drawn with a fixed seed (the same on every MPI rank), on which most candidates
    # fail; the rest are matched in full: 12 per species at most
    draw = np.random.default_rng(0)
    sample = np.unique(
        np.concatenate([draw.choice(indices, size=min(12, indices.size), replace=False) for indices in members.values()])
    )

    def plausible(matrix_fraction, offsets):
        """The indices (in order) of the ``offsets`` (fractional) whose ``f -> f N + t`` carries every sampled atom
        within the tolerance onto an atom of its species: a query per sampled atom for the offsets still in."""

        keep = np.arange(len(offsets))
        for atom in sample:
            if not keep.size:
                break
            symbol = symbols[atom]
            images = fractions[atom] @ matrix_fraction + offsets[keep]
            queries = np.mod(images, 1.0)
            queries[queries >= 1.0] = 0.0
            _, nearest = trees[symbol].query(queries, k=1)
            difference = images - fractions[members[symbol][nearest]]
            difference -= np.round(difference)
            keep = keep[np.linalg.norm(difference @ vectors, axis=1) <= tolerance]
        return keep

    def operation(matrix_fraction, offset):
        """``(offset, image)`` where ``f -> f N + t`` maps every atom (a permutation of each species), else ``None``."""

        image = _cell_matches(fractions, symbols, trees, members, vectors, fractions @ matrix_fraction + offset,
                              tolerance)
        if image is None or np.unique(image).size != image.size:
            return None
        return np.asarray(offset, dtype=np.float64), image

    def anchor_orbit(tables):
        """The atoms the translations generated by ``tables`` carry the anchor onto (a mask)."""

        seen = np.zeros(len(symbols), dtype=bool)
        seen[anchor] = True
        frontier = np.asarray([anchor], dtype=np.int64)
        while frontier.size:
            reached = np.unique(np.concatenate([table[frontier] for table in tables]))
            frontier = reached[~seen[reached]]
            seen[frontier] = True
        return seen

    identity = np.eye(3)
    # The pure translations T act freely on the atoms: each is known by the atom it carries the anchor to.  Only the
    # generators are matched against the atoms and kept (a table of N atoms each); T is Abelian.
    shifts = np.mod(fractions[members[rare]] - fractions[anchor], 1.0)
    generators: list[np.ndarray] = []
    generator_offsets: list[np.ndarray] = []
    reached = np.zeros(len(symbols), dtype=bool)
    reached[anchor] = True
    for index in plausible(identity, shifts):
        if reached[int(members[rare][index])]:
            continue
        found = operation(identity, shifts[index])
        if found is None:
            continue
        generators.append(found[1])
        generator_offsets.append(found[0] @ vectors)
        reached = anchor_orbit(generators)
    translation_order = int(np.count_nonzero(reached))
    representatives = []
    fractional_matrices = []
    for rotation, matrix_fraction in _lattice_point_operations(vectors):
        if np.array_equal(matrix_fraction, identity):
            representatives.append((np.asarray(rotation, dtype=np.float64), np.zeros(3), np.arange(len(symbols))))
            fractional_matrices.append(np.asarray(matrix_fraction, dtype=np.float64))
            continue
        # the translations of an operation R are one coset t_R + T: the first that maps the atoms represents it
        # (in chunks of the candidates, which stop at the first that maps every atom)
        offsets = np.mod(fractions[members[rare]] - fractions[anchor] @ matrix_fraction, 1.0)
        found = None
        for begin in range(0, len(offsets), _CANDIDATE_CHUNK):
            for index in plausible(matrix_fraction, offsets[begin : begin + _CANDIDATE_CHUNK]) + begin:
                found = operation(matrix_fraction, offsets[index])
                if found is not None:
                    break
            if found is not None:
                # f -> f N + t is r -> R r + t A (rows of A the lattice vectors)
                representatives.append((np.asarray(rotation, dtype=np.float64), found[0] @ vectors, found[1]))
                fractional_matrices.append(np.asarray(matrix_fraction, dtype=np.float64))
                break
    order = len(representatives) * translation_order
    if order <= 1:
        return positions.copy(), 1, 0, 0.0
    group_error = (
        f"{SYMMETRIZE_POSITIONS_SWITCH}=on: the {order} operations of the cell that map the atoms within "
        f"{tolerance:g} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE) form no group within it, so their average is not a "
        "symmetric structure near the input; lower the tolerance"
    )
    # The average over G = T (representatives): over T first, which gives positions T leaves invariant, then over the
    # representatives of its cosets, which for such positions is the average over G (the mean translations compose
    # as the operations do; T is normal in G).
    mismatch = 0.0
    averaged = positions
    pure_translations = np.zeros((1, 3))
    if generators:
        means, mismatch = _generator_translations(positions, vectors, generators, generator_offsets)
        representative, translation = _translation_orbits(vectors, generators, means)
        # T acts freely: every orbit has |T| atoms, and the anchor's lists the translations of T
        if np.any(np.bincount(representative)[np.unique(representative)] != translation_order):
            raise ValueError(group_error)
        averaged = _translation_average(positions, vectors, representative, translation)
        pure_translations = translation[representative == representative[anchor]]
    symmetric = _cell_average(averaged, vectors, representatives)
    pinned, shift = 0, np.zeros(3)
    if grid is not None:
        shift, pinned = _grid_shift(
            symmetric, vectors, representatives, fractional_matrices, pure_translations, grid, tolerance
        )
        symmetric = symmetric - shift
    scale = max(1.0, float(np.max(np.abs(vectors))))
    error = _cell_invariance_error(symmetric, vectors, representatives)
    if generators:
        error = max(error, _generator_translations(symmetric, vectors, generators, generator_offsets)[1])
    largest_move = float(np.max(np.linalg.norm(symmetric - positions, axis=1), initial=0.0))
    if error > _INVARIANCE_TOLERANCE * scale or mismatch > 2.0 * tolerance or largest_move > 2.0 * tolerance:
        raise ValueError(group_error)
    return symmetric, order, pinned, float(np.linalg.norm(shift))


def symmetrized_cell_positions(positions, symbols: Sequence[str], lattice_vectors, tolerance: float, grid=None):
    """``(positions, order)``: the average of ``positions`` over the operations ``r -> R r + t`` of a bulk cell that
    map every atom within ``tolerance`` bohr onto an atom of its species modulo the lattice (``R`` an orthogonal map
    of the lattice onto itself, ``t`` any translation), shifted as a whole onto the ``grid`` ``(counts, shift)``
    (points ``(n + shift)/counts`` in fractional coordinates; see the module) where one is given.  Raises
    ``ValueError`` where they form no group."""

    symmetric, order, _, _ = _symmetrized_cell(positions, symbols, lattice_vectors, tolerance, grid)
    return symmetric, order


def _cell_grid(problem, cell):
    """``(counts, shift)`` of the grid the preparation builds for a bulk cell (the default or the Fortran-derived
    routines), or ``None`` where the settings give none."""

    settings = getattr(problem, "grid", None)
    if settings is None or not hasattr(settings, "spacing"):
        return None
    from ..Grid.pbc import default_periodic_grid_counts, fortran_grid_counts

    try:
        if getattr(settings, "fortran_routines", False):
            counts = fortran_grid_counts(cell, settings)[0]
        else:
            counts = default_periodic_grid_counts(cell, settings)
    except (ValueError, TypeError, AttributeError):
        return None
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size != 3 or np.any(counts < 1):
        return None
    return counts, np.asarray(getattr(settings, "shift", (0.0, 0.0, 0.0)), dtype=np.float64)


def symmetrize_input_positions(translation):
    """``(translation, notes)``: the translation with its positions symmetrized under
    ``PARSEC_SYMMETRIZE_POSITIONS=on`` (see the module), the note for parsec.out; unchanged and no note under ``off``.

    Raises ``ValueError`` for an unknown value of the switch or of ``PARSEC_SYMMETRY_ATOM_TOLERANCE``, and where the
    operations found form no group.
    """

    if symmetrize_positions_setting() == "off":
        return translation, []
    from ..acceleration.Symmetry.axis_reflection import symmetry_atom_tolerance

    tolerance = symmetry_atom_tolerance()
    problem = translation.problem
    atoms = tuple(problem.atoms)
    if not atoms:
        return translation, []
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    symbols = [atom.symbol for atom in atoms]
    cell = getattr(problem, "periodic_cell", None)
    pinned, shift = 0, 0.0
    if cell is not None:
        if int(getattr(problem.grid, "periodicity", 3)) != 3 or int(getattr(cell, "dimensions", 3)) != 3:
            return translation, [
                f"NOTE: {SYMMETRIZE_POSITIONS_SWITCH}=on: the positions of a wire or slab are kept (bulk cells and "
                "clusters only)"
            ]
        symmetric, order, pinned, shift = _symmetrized_cell(
            positions, symbols, cell.lattice_vectors, tolerance, _cell_grid(problem, cell)
        )
        kind = "cell"
    else:
        from ..V_ion import center_cluster_geometry

        centred = (
            np.asarray([atom.position for atom in center_cluster_geometry(atoms)], dtype=np.float64)
            if getattr(problem, "recenter_geometry", False)
            else positions
        )
        offset = positions - centred
        symmetric, order = symmetrized_cluster_positions(centred, symbols, tolerance)
        symmetric = symmetric + offset
        kind = "cluster"
    report = PositionSymmetrization(
        order=order,
        tolerance=tolerance,
        largest_move=float(np.max(np.linalg.norm(symmetric - positions, axis=1), initial=0.0)),
        kind=kind,
        pinned=pinned,
        shift=shift,
    )
    if order <= 1:
        return translation, [report.note()]
    moved = tuple(replace(atom, position=symmetric[index].copy()) for index, atom in enumerate(atoms))
    return replace(translation, problem=replace(problem, atoms=moved)), [report.note()]


__all__ = [
    "PositionSymmetrization",
    "SYMMETRIZE_POSITIONS_SWITCH",
    "symmetrize_input_positions",
    "symmetrize_positions_setting",
    "symmetrized_cell_positions",
    "symmetrized_cluster_positions",
]
