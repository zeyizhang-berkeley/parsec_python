"""Opt-in standard orientation of a cluster (``PARSEC_STANDARD_ORIENTATION``).

PARSEC looks for the point group of a cluster among the 48 signed permutations of the Cartesian axes, about the
origin, after the bounding-box recentring of ``initial.f90`` (only beyond 0.1 bohr, and not in a restart)
(:func:`parsec_python.Forces.symmetry.atom_symmetry_operations`, the sectors of
:mod:`parsec_python.acceleration.Symmetry.axis_reflection`).  A molecule whose symmetry axes are not along the
coordinate axes, or that sits up to 0.1 bohr off its centre, keeps a smaller group or none: methane rotated by 10
degrees about z keeps four operations (two of them commuting) instead of 24, methane 0.05 bohr off the origin four.

With ``PARSEC_STANDARD_ORIENTATION=on`` the input reader looks for the frame in which the atoms, placed as PARSEC
places them, have their largest group of signed permutations, and runs the calculation there:

1. the centre is the centroid of the atoms, which every symmetry operation keeps;
2. the two-fold axes and the mirror normals of the cluster are found among the directions of the atoms, the sums,
   differences and cross products of pairs of atoms of its smallest shells and the principal axes of inertia;
3. every frame whose coordinate axes are one such axis, or two orthogonal ones, is scored by the group of signed
   permutations it gives: first the order of its largest group of commuting involutions, which the abelian sectors of
   the code use, then the order of the whole group (weighted sectors); of the best frames the one closest to the
   input wins.  In a relaxation with ``Move_Flag`` a frame counts as the run decides
   (:func:`parsec_python.Relaxation.run.symmetry_decision`): its group is that of the species; where an operation of
   that group exchanges a fixed atom with a movable one, a relaxation by an optimizer keeps the subgroup of the
   operations that map the fixed atoms onto fixed atoms (``PARSEC_MOVE_FLAG_SUBGROUP=auto``, since 2026-11-20 also in
   this score), and under ``off`` (or for other moves) the frame counts as no group at all;
4. the frame is refined (a Kabsch fit to the symmetrized structure) and the atoms are symmetrized exactly over the
   group (an orbit average, the representative averaged over its stabilizer with exactly rounded sums), then moved
   so that their bounding box is centred at the origin, as PARSEC centres a cluster.

Nothing changes unless the group grows: an aligned, centred input keeps its coordinates bit for bit.  The atoms
move by the symmetrization up to ``PARSEC_STANDARD_ORIENTATION_TOLERANCE`` (1e-3 bohr by default), the distance
within which an operation is accepted; the move is reported.  The calculation, its grid and the files of the grid
are in the new frame; coordinates, forces, the dipole and the files of a relaxation are reported in the frame of
the input (:class:`StandardOrientation`).  Energies and forces differ from those of the input orientation by the
egg-box effect of the grid (the atoms sit elsewhere between the grid points): measured on A100 1-2.4e-4 Ry and
1-2e-3 Ry/bohr for methane on a grid of 0.25 bohr, 2.1e-2 Ry and 4e-2 Ry/bohr for C87H76 on 0.4 bohr; the
calculation frame reproduces the aligned input to the last printed digit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, permutations, product
import math
import os
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


SWITCH = "PARSEC_STANDARD_ORIENTATION"
TOLERANCE_SWITCH = "PARSEC_STANDARD_ORIENTATION_TOLERANCE"
#: the distance (bohr) within which an operation must carry every atom onto an atom of its species, by default
DEFAULT_TOLERANCE = 1.0e-3
#: the atom tolerance of the detection of the accelerated route
#: (``acceleration.Symmetry.axis_reflection.DEFAULT_ATOM_TOLERANCE``) unless ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` says
#: otherwise: the group the run would use without the switch
DETECTION_TOLERANCE = 1.0e-7
#: at most this many atom pairs of the smallest shells give the candidate axes
_PAIR_BUDGET = 4000


def standard_orientation_setting() -> str:
    """``PARSEC_STANDARD_ORIENTATION``: ``off`` (default) keeps the atoms as PARSEC places them; ``on`` runs a cluster
    in the frame and about the centre of its largest group of signed permutations (module docstring)."""

    raw = os.environ.get(SWITCH, "off").strip().lower() or "off"
    if raw not in {"off", "on"}:
        raise ValueError(f"{SWITCH} must be off or on, not {raw!r}")
    return raw


def standard_orientation_tolerance() -> float:
    """``PARSEC_STANDARD_ORIENTATION_TOLERANCE``: the distance (bohr) within which an operation must carry every atom
    onto an atom of its species to be kept by the standard orientation, ``1e-3`` by default (above 0, at most 0.01).
    The atoms are symmetrized over the group found, so that they move by up to about this distance."""

    raw = os.environ.get(TOLERANCE_SWITCH, "").strip()
    if not raw:
        return DEFAULT_TOLERANCE
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{TOLERANCE_SWITCH} must be a distance in bohr, not {raw!r}") from None
    if not (np.isfinite(value) and 0.0 < value <= 0.01):
        raise ValueError(f"{TOLERANCE_SWITCH} must be above 0 and at most 0.01 bohr, not {raw!r}")
    return value


def detection_tolerance() -> float:
    """The atom tolerance of the group detection of the run (``PARSEC_SYMMETRY_ATOM_TOLERANCE``, default 1e-7)."""

    from ..acceleration.Symmetry.axis_reflection import symmetry_atom_tolerance

    return symmetry_atom_tolerance()


@dataclass(frozen=True)
class StandardOrientation:
    """The frame a cluster is calculated in, against the frame of the input as PARSEC places it.

    ``rotation`` ``Q`` is proper; its columns are the axes of the calculation frame in the reference frame, and
    ``centre`` ``d`` is the origin of the calculation frame there: ``x_calc = Q^T (x_ref - d)``,
    ``x_ref = Q x_calc + d``.  The reference frame is that of the run without the switch: the coordinates of the
    input moved by ``recentring`` (PARSEC's bounding-box shift, zero where it did not apply).
    ``abelian_before``/``order_before`` are the orders of the largest group of commuting involutions (the abelian
    sectors) and of the whole group of signed permutations the run would use in the reference frame, ``abelian_after``/``order_after`` those of the
    calculation frame; ``displacement`` is the largest move of an atom by the symmetrization (bohr).
    """

    rotation: np.ndarray
    centre: np.ndarray
    recentring: np.ndarray
    abelian_before: int
    order_before: int
    abelian_after: int
    order_after: int
    displacement: float
    tolerance: float

    def to_calculation(self, positions) -> np.ndarray:
        """Positions ``(..., 3)`` of the reference frame in the calculation frame."""

        return (np.asarray(positions, dtype=np.float64) - self.centre) @ self.rotation

    def to_reference(self, positions) -> np.ndarray:
        """Positions ``(..., 3)`` of the calculation frame in the reference frame."""

        return np.asarray(positions, dtype=np.float64) @ self.rotation.T + self.centre

    def vectors_to_reference(self, vectors) -> np.ndarray:
        """Vectors ``(..., 3)`` (forces, displacements) of the calculation frame in the reference frame."""

        return np.asarray(vectors, dtype=np.float64) @ self.rotation.T

    def restart_records(self) -> tuple[str, ...]:
        """The records of the map that ``relax_restart.dat`` keeps after its header
        (:data:`parsec_python.Relaxation.files.ORIENTATION_HEADER`): the rows of ``Q``, ``d``, the recentring, the
        group orders, the largest move and the tolerance, with decimals that read back to the same bits."""

        def row(values) -> str:
            return " " + " ".join(repr(float(value)) for value in values)

        return (
            *(row(line) for line in self.rotation),
            row(self.centre),
            row(self.recentring),
            " "
            + " ".join(
                str(int(value))
                for value in (self.abelian_before, self.order_before, self.abelian_after, self.order_after)
            ),
            row((self.displacement, self.tolerance)),
        )

    @classmethod
    def from_restart_records(cls, records) -> "StandardOrientation":
        """The map of :meth:`restart_records`; a ``ValueError`` where the records do not hold one."""

        try:
            numbers = [[float(token) for token in record.split()] for record in records]
        except ValueError:
            raise ValueError("relax_restart.dat: its standard orientation block holds no numbers") from None
        if [len(values) for values in numbers] != [3, 3, 3, 3, 3, 4, 2]:
            raise ValueError("relax_restart.dat: its standard orientation block is incomplete")
        rotation = np.asarray(numbers[:3], dtype=np.float64)
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-10) or np.linalg.det(rotation) <= 0.0:
            raise ValueError("relax_restart.dat: its standard orientation block holds no rotation")
        orders = [int(value) for value in numbers[5]]
        return cls(
            rotation=rotation,
            centre=np.asarray(numbers[3], dtype=np.float64),
            recentring=np.asarray(numbers[4], dtype=np.float64),
            abelian_before=orders[0],
            order_before=orders[1],
            abelian_after=orders[2],
            order_after=orders[3],
            displacement=numbers[6][0],
            tolerance=numbers[6][1],
        )

    def note(self) -> str:
        """The note of the input reader."""

        return (
            f"{SWITCH}=on: the cluster is calculated in the frame of its largest group of signed permutations "
            f"(abelian sector group {self.abelian_before} -> {self.abelian_after}, signed permutations "
            f"{self.order_before} -> {self.order_after}); its atoms moved by up to {self.displacement:.2e} bohr when "
            "they were symmetrized over that group. The SCF, the grid, the files of the grid (cube, parsec.dat, "
            "wfn.dat, the .npz archive) and relax_restart.dat are in that frame (a Restart_Run with the switch on "
            "and the same input finds it again; relax_restart.dat records it, and a restart from it reports in the "
            "frame of the run that wrote it); coordinates, forces, the dipole, atom.cor, relaxation.xyz and the "
            "forces and final positions of the MPI runner's JSON report are in the frame of the input as PARSEC "
            "places it. "
            "Energies and forces differ from those of the input orientation by the egg-box effect of the grid "
            "(the atoms sit elsewhere between the grid points)."
        )

    def report_lines(self) -> list[str]:
        """The block of ``parsec.out``: the map between the frames."""

        lines = [
            "",
            f" Standard orientation ({SWITCH}=on):",
            " the calculation frame is x_calc = Q^T (x - d), x in the frame of the input",
            f" abelian sector group {self.abelian_before:2d} -> {self.abelian_after:2d}, "
            f"signed permutations {self.order_before:2d} -> {self.order_after:2d}",
            f" largest move of an atom by the symmetrization [bohr] = {self.displacement:12.4e}",
            " Q (columns: the axes of the calculation frame):",
        ]
        lines += [" " + "".join(f"{value:16.10f}" for value in row) for row in self.rotation]
        lines.append(" d [bohr] = " + "".join(f"{value:16.10f}" for value in self.centre))
        if np.any(self.recentring != 0.0):
            lines.append(
                " (the frame of the input is moved by PARSEC's recentring: "
                + " ".join(f"{-value:.6f}" for value in self.recentring)
                + " bohr)"
            )
        lines += [
            " Coordinates, forces and the dipole below are in the frame of the input;",
            " the grid and its files are in the calculation frame.",
            "",
        ]
        return lines


# ------------------------------------------------------------------ helpers


def _signed_permutations() -> np.ndarray:
    """The 48 signed permutation matrices (integer), identity first."""

    matrices = []
    for order in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            matrices.append(matrix)
    return np.asarray(matrices)


_SIGNED = _signed_permutations()
_PROPER = _SIGNED[[int(round(np.linalg.det(matrix))) == 1 for matrix in _SIGNED]]


def _product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``left @ right`` summed in a fixed order (not by a threaded BLAS), so that every MPI rank finds the same bits."""

    return np.einsum("ij,jk->ik", left, right, optimize=False)


def _grid_operations(grid_shift) -> np.ndarray:
    """The signed permutations that map a grid shifted by ``grid_shift`` steps onto itself."""

    if grid_shift is None:
        return _SIGNED
    shift = np.asarray(grid_shift, dtype=np.float64)
    keep = []
    for matrix in _SIGNED:
        offset = matrix @ shift - shift
        keep.append(bool(np.max(np.abs(offset - np.rint(offset))) <= 1.0e-9))
    return _SIGNED[np.asarray(keep)]


class _Matcher:
    """Nearest atoms of each species, for the test of an operation about the origin."""

    def __init__(self, positions: np.ndarray, labels: Sequence) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        groups: dict = {}
        for index, label in enumerate(labels):
            groups.setdefault(label, []).append(index)
        self.groups = [np.asarray(members, dtype=np.int64) for members in groups.values()]
        self.trees = [cKDTree(self.positions[members]) for members in self.groups]
        self.label_of = np.empty(len(labels), dtype=np.int64)
        for number, members in enumerate(self.groups):
            self.label_of[members] = number

    def image(self, matrix, tolerance: float, subset: np.ndarray | None = None) -> np.ndarray | None:
        """The atom each atom goes to under ``matrix`` (acting on column vectors), or ``None``.

        With ``subset`` only those atoms are tested (a quick test; no image is returned, only ``True``-like).
        """

        matrix = np.asarray(matrix, dtype=np.float64)
        if subset is not None:
            rotated = self.positions[subset] @ matrix.T
            for number in np.unique(self.label_of[subset]):
                chosen = self.label_of[subset] == number
                distance, _ = self.trees[number].query(rotated[chosen], distance_upper_bound=tolerance)
                if not np.all(np.isfinite(distance)):
                    return None
            return np.empty(0, dtype=np.int64)
        rotated = self.positions @ matrix.T
        image = np.empty(self.positions.shape[0], dtype=np.int64)
        for members, tree in zip(self.groups, self.trees):
            distance, nearest = tree.query(rotated[members], distance_upper_bound=tolerance)
            if not np.all(np.isfinite(distance)) or np.unique(nearest).size != members.size:
                return None
            image[members] = members[nearest]
        return image


def _group(
    matcher: _Matcher,
    frame: np.ndarray,
    operations: np.ndarray,
    tolerance: float,
    subset: np.ndarray | None = None,
):
    """The signed permutations of the frame (columns of ``frame``) that map the atoms, with their atom images.

    ``subset`` names atoms tested first (a quick rejection on a few atoms before all of them).
    """

    found = []
    for matrix in operations:
        rotation = frame @ matrix @ frame.T
        if subset is not None and subset.size and matcher.image(rotation, tolerance, subset) is None:
            continue
        image = matcher.image(rotation, tolerance)
        if image is not None:
            found.append((matrix, image))
    return found


def _abelian_order(matrices) -> int:
    """The order of the largest group of commuting involutions among ``matrices`` (a group of signed permutations).

    The abelian sectors of the accelerated route take that group
    (``acceleration.Symmetry.axis_reflection.SignedPermutationReduction.detect``: the largest elementary abelian
    subgroup ``C2^k`` of the signed permutations of the atoms, not only the diagonal one).
    """

    identity = np.eye(3, dtype=np.int64)
    involutions = [
        np.asarray(matrix, dtype=np.int64)
        for matrix in matrices
        if not np.array_equal(matrix, identity) and np.array_equal(matrix @ matrix, identity)
    ]
    for rank in range(3, 0, -1):
        for generators in combinations(involutions, rank):
            if any(not np.array_equal(a @ b, b @ a) for a, b in combinations(generators, 2)):
                continue
            products = set()
            for exponents in product((0, 1), repeat=rank):
                operation = identity
                for enabled, generator in zip(exponents, generators):
                    if enabled:
                        operation = operation @ generator
                products.add(operation.tobytes())
            if len(products) == 2**rank:
                return 2**rank
    return 1


def _run_group(
    matcher: _Matcher,
    frame: np.ndarray,
    operations: np.ndarray,
    tolerance: float,
    subset: np.ndarray | None = None,
    movable: np.ndarray | None = None,
    subgroup: bool = False,
):
    """The group the run would use in the frame, or ``None`` where it would use none.

    Without ``movable`` this is :func:`_group`.  With the movable flags of a relaxation it follows
    :func:`parsec_python.Relaxation.run.symmetry_decision`: where the group of the species among all 48 signed
    permutations maps the movable atoms onto movable ones (:func:`parsec_python.Forces.symmetry.operations_preserving`)
    the group used is that one, restricted to ``operations`` (the grid); where it does not, ``subgroup`` (a relaxation
    by an optimizer under ``PARSEC_MOVE_FLAG_SUBGROUP=auto``) keeps the subgroup of the operations that map the fixed
    atoms onto fixed atoms where it is more than the identity, and the relaxation uses no symmetry otherwise.  That
    test takes at least the atom tolerance of ``symmetry_decision``
    (:data:`parsec_python.Forces.symmetry.POSITION_TOLERANCE`).
    """

    if movable is None:
        return _group(matcher, frame, operations, tolerance, subset)
    from ..Forces.symmetry import POSITION_TOLERANCE

    check = max(float(tolerance), POSITION_TOLERANCE)
    whole = _group(matcher, frame, _SIGNED, check, subset)
    breaking = any(not np.array_equal(movable[image], movable) for _, image in whole)
    if breaking and not subgroup:
        return None
    if check != tolerance:
        found = _group(matcher, frame, operations, tolerance, subset)
    elif operations is _SIGNED:
        found = whole
    else:
        kept = {np.asarray(matrix, dtype=np.int64).tobytes() for matrix in operations}
        found = [(matrix, image) for matrix, image in whole if np.asarray(matrix, dtype=np.int64).tobytes() in kept]
    if not breaking:
        return found
    # PARSEC_MOVE_FLAG_SUBGROUP=auto: the operations that map the fixed atoms onto fixed atoms
    preserving = [(matrix, image) for matrix, image in found if np.array_equal(movable[image], movable)]
    return preserving if len(preserving) > 1 else None


def _score(group) -> tuple[int, int]:
    """The group of the abelian sectors, then the whole group of signed permutations (weighted sectors).

    ``None`` (a frame where the run would use no group) scores as the identity alone.
    """

    if group is None:
        return (1, 1)
    return (_abelian_order([matrix for matrix, _ in group]), len(group))


def _unit(vectors: np.ndarray, floor: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > floor
    return vectors[keep] / norms[keep, None]


def _distinct_directions(directions: np.ndarray, angle: float) -> np.ndarray:
    """One of every set of directions within ``angle`` of each other (up to sign), in the order given."""

    if directions.shape[0] == 0:
        return directions
    tree = cKDTree(np.vstack([directions, -directions]))
    count = directions.shape[0]
    taken = np.zeros(count, dtype=bool)
    kept = []
    for index in range(count):
        if taken[index]:
            continue
        kept.append(index)
        for neighbour in tree.query_ball_point(directions[index], angle):
            taken[neighbour % count] = True
    return directions[kept]


def _shells(radii: np.ndarray, labels: Sequence, gap: float) -> list[np.ndarray]:
    """Atoms of one species at one distance from the centre (within ``gap``), smallest sets first."""

    shells = []
    by_label: dict = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    for members in by_label.values():
        members = np.asarray(members, dtype=np.int64)
        members = members[np.argsort(radii[members], kind="stable")]
        start = 0
        for stop in range(1, members.size + 1):
            if stop == members.size or radii[members[stop]] - radii[members[stop - 1]] > gap:
                shells.append(members[start:stop])
                start = stop
    shells.sort(key=lambda shell: (shell.size, -float(radii[shell].mean()), int(shell[0])))
    return shells


def _candidate_directions(y: np.ndarray, labels: Sequence, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Directions that may be two-fold axes or mirror normals, and the atoms of the shells they came from."""

    radii = np.linalg.norm(y, axis=1)
    outer = radii > max(10.0 * tolerance, 1.0e-6)
    directions = []
    # principal axes of inertia, of all atoms and of every species
    for chosen in [np.ones(len(labels), dtype=bool)] + [
        np.asarray([value == label for value in labels]) for label in dict.fromkeys(labels)
    ]:
        points = y[chosen]
        tensor = np.sum(points * points) * np.eye(3) - _product(points.T, points)
        directions.extend(np.linalg.eigh(tensor)[1].T)
    shells = [shell for shell in _shells(radii, labels, 2.0 * tolerance) if outer[shell[0]]]
    used: list[np.ndarray] = []
    pairs = 0
    for shell in shells:
        count = shell.size * (shell.size - 1) // 2
        if used and (pairs + count > _PAIR_BUDGET or len(used) >= 4):
            break
        used.append(shell)
        pairs += count
    for shell in used:
        points = y[shell]
        directions.extend(points)
        first, second = np.triu_indices(points.shape[0], k=1)
        if first.size:
            directions.extend(points[first] + points[second])
            directions.extend(points[first] - points[second])
            directions.extend(np.cross(points[first], points[second]))
    scale = max(float(np.max(radii)), 1.0)
    units = _unit(np.asarray(directions, dtype=np.float64), 1.0e-8 * scale * scale)
    subset = np.concatenate(used) if used else np.flatnonzero(outer)
    return _distinct_directions(units, 1.0e-3), subset


def _refined_axis(y: np.ndarray, image: np.ndarray, mirror: bool) -> np.ndarray:
    """The axis that fits the pairs of an accepted two-fold rotation (sums) or mirror (differences) best."""

    vectors = y - y[image] if mirror else y + y[image]
    values, axes = np.linalg.eigh(_product(vectors.T, vectors))
    return axes[:, -1]


def _minimal_rotation(target: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The rotation of smallest angle that carries the unit vector ``target`` onto the unit vector ``axis``."""

    cosine = float(np.clip(target @ axis, -1.0, 1.0))
    cross = np.cross(target, axis)
    sine = float(np.linalg.norm(cross))
    if sine < 1.0e-15:
        return np.eye(3)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _closest_to_identity(frame: np.ndarray) -> np.ndarray:
    """``frame P`` for the proper signed permutation ``P`` that brings it closest to the identity."""

    traces = [float(np.trace(frame @ matrix)) for matrix in _PROPER]
    return frame @ _PROPER[int(np.argmax(traces))]


def _candidate_frames(axes: np.ndarray) -> list[np.ndarray]:
    frames = [np.eye(3)]
    for axis in axes:
        target = np.zeros(3)
        component = int(np.argmax(np.abs(axis)))
        target[component] = 1.0 if axis[component] > 0.0 else -1.0
        frames.append(_minimal_rotation(target, axis))
    for first in range(axes.shape[0]):
        for second in range(first + 1, axes.shape[0]):
            u = axes[first]
            v = axes[second]
            if abs(float(u @ v)) > 1.0e-2:
                continue
            v = v - (u @ v) * u
            v = v / np.linalg.norm(v)
            frames.append(np.column_stack([u, v, np.cross(u, v)]))
    distinct: list[np.ndarray] = []
    for frame in frames:
        frame = _closest_to_identity(frame)
        if not any(np.max(np.abs(frame - other)) < 1.0e-6 for other in distinct):
            distinct.append(frame)
    return distinct


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The proper rotation ``R`` that minimizes ``sum |R source_i - target_i|^2``."""

    covariance = _product(target.T, source)
    left, _, right = np.linalg.svd(covariance)
    sign = 1.0 if np.linalg.det(left @ right) > 0.0 else -1.0
    return left @ np.diag([1.0, 1.0, sign]) @ right


def _orbit_average(z: np.ndarray, group) -> np.ndarray:
    """``(1/|G|) sum_P P^T z[image_P]``: the nearest configuration with the symmetry (not exact in round-off)."""

    total = np.zeros_like(z)
    for matrix, image in group:
        total += z[image] @ matrix.astype(np.float64)
    return total / float(len(group))


def _exactly_symmetric(z: np.ndarray, group) -> np.ndarray:
    """Positions that the signed permutations of ``group`` map onto each other bit for bit.

    One representative per orbit takes the orbit average, averaged over its stabilizer with exactly rounded sums
    (so that the components a stabilizer swaps are equal and those it negates are zero, bit for bit); every other
    atom of the orbit is the image of the representative under a signed permutation, which only moves and negates
    components.
    """

    average = _orbit_average(z, group)
    count = z.shape[0]
    result = np.empty_like(z)
    done = np.zeros(count, dtype=bool)
    matrices = [matrix.astype(np.float64) for matrix, _ in group]
    images = [image for _, image in group]
    for representative in range(count):
        if done[representative]:
            continue
        stabilizer = [m for m, image in zip(matrices, images) if image[representative] == representative]
        candidates = np.asarray([m @ average[representative] for m in stabilizer])
        point = np.array(
            [math.fsum(candidates[:, axis]) / len(stabilizer) for axis in range(3)], dtype=np.float64
        )
        for matrix, image in zip(matrices, images):
            target = int(image[representative])
            if not done[target]:
                result[target] = matrix @ point
                done[target] = True
    return result


# ------------------------------------------------------------------ search


@dataclass(frozen=True)
class OrientationSearch:
    """What :func:`find_standard_orientation` found: the orientation (``None`` where the group does not grow) and
    the positions of the calculation frame, with the scores of both frames."""

    orientation: StandardOrientation | None
    positions: np.ndarray | None
    before: tuple[int, int]
    after: tuple[int, int]


def find_standard_orientation(
    positions,
    labels: Sequence,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    detection: float = DETECTION_TOLERANCE,
    grid_shift=None,
    recentring=None,
    movable=None,
    move_flag_subgroup: bool | None = None,
) -> OrientationSearch:
    """The frame and centre of the largest group of signed permutations of atoms at ``positions``.

    ``positions`` ``(atoms, 3)`` are in the reference frame (the input as PARSEC places it), ``labels`` say which
    atoms an operation may exchange (the species).  ``detection`` is the tolerance of the
    run's own detection, which gives the group of the reference frame; ``grid_shift`` (in steps) keeps the
    operations that map the grid onto itself.  ``movable`` (the movable flags of a relaxation, or ``None``) makes a
    frame count as the relaxation decides (:func:`_run_group`): a frame whose group exchanges a fixed atom with a
    movable one counts by the subgroup of the operations that map the fixed atoms onto fixed atoms where
    ``move_flag_subgroup`` (a relaxation by an optimizer under ``PARSEC_MOVE_FLAG_SUBGROUP=auto``; ``None`` reads that
    switch), else as the identity alone.  The positions of the calculation frame are exactly symmetric under the
    group found.
    """

    reference = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    labels = list(labels)
    operations = _grid_operations(grid_shift)
    recentring = np.zeros(3) if recentring is None else np.asarray(recentring, dtype=np.float64)
    mask = None if movable is None else np.asarray(movable, dtype=bool).reshape(-1)
    if move_flag_subgroup is None:
        from ..Relaxation.run import move_flag_subgroup_setting

        move_flag_subgroup = move_flag_subgroup_setting() == "auto"
    subgroup = bool(move_flag_subgroup)
    before_matcher = _Matcher(reference, labels)
    before = _score(_run_group(before_matcher, np.eye(3), operations, detection, movable=mask, subgroup=subgroup))
    if before == _score([(matrix, None) for matrix in operations]):
        return OrientationSearch(None, None, before, before)
    # (a single atom, or atoms that all sit at their centroid, keeps every operation about it: the identity frame)
    centroid = reference.mean(axis=0)
    y = reference - centroid
    loose = 10.0 * tolerance
    matcher = _Matcher(y, labels)
    directions, subset = _candidate_directions(y, labels, tolerance)
    axes = []
    for direction in directions:
        for mirror in (False, True):
            outer = np.outer(direction, direction)
            matrix = (np.eye(3) - 2.0 * outer) if mirror else (2.0 * outer - np.eye(3))
            if subset.size and matcher.image(matrix, loose, subset) is None:
                continue
            image = matcher.image(matrix, loose)
            if image is None:
                continue
            axis = _refined_axis(y, image, mirror)
            axes.append(axis / np.linalg.norm(axis))
    axes = _distinct_directions(np.asarray(axes, dtype=np.float64).reshape(-1, 3), 1.0e-3)
    scored = []
    for number, frame in enumerate(_candidate_frames(axes)):
        group = _run_group(matcher, frame, operations, 2.0 * tolerance, subset, movable=mask, subgroup=subgroup)
        score = _score(group)
        if score > before:
            # the best score first; of equal ones the frame nearest the input (largest trace), then the first found
            scored.append((score, float(np.trace(frame)), -number, frame, group))
    scored.sort(key=lambda item: (item[0], round(item[1], 9), item[2]), reverse=True)
    last = before
    for score, _, _, frame, group in scored:
        found = _symmetrized_frame(y, frame, group, score, labels, operations, detection, mask, subgroup)
        if found is None:
            continue
        frame, calculation, middle = found
        orientation = StandardOrientation(
            rotation=frame,
            centre=centroid + frame @ middle,
            recentring=recentring,
            abelian_before=before[0],
            order_before=before[1],
            abelian_after=score[0],
            order_after=score[1],
            displacement=0.0,
            tolerance=tolerance,
        )
        displacement = float(
            np.max(np.linalg.norm(orientation.to_reference(calculation) - reference, axis=1))
        )
        last = score
        if displacement <= tolerance:
            return OrientationSearch(
                replace(orientation, displacement=displacement), calculation, before, score
            )
    return OrientationSearch(None, None, before, last if scored else before)


def _symmetrized_frame(y, frame, group, score, labels, operations, detection, movable=None, subgroup=False):
    """The frame refined against the symmetrized structure and the structure symmetrized exactly in it.

    Returns ``(frame, positions, middle)`` with the positions centred on their bounding box (``middle`` is the
    shift taken off), or ``None`` where the symmetrized structure does not keep the group at the tolerance of
    the detection (the operations found at the tolerance do not form a group).
    """

    z = _product(y, frame)
    for _ in range(4):
        target = _orbit_average(z, group)
        singular = np.linalg.svd(_product(target.T, z), compute_uv=False)
        if singular[1] <= 1.0e-12 * max(singular[0], 1.0e-300):
            # a linear molecule: a rotation about its axis is free, and changes nothing
            break
        rotation = _kabsch(z, target)
        z = _product(z, rotation.T)
        frame = frame @ rotation.T
    symmetric = _exactly_symmetric(z, group)
    middle = 0.5 * (symmetric.min(axis=0) + symmetric.max(axis=0))
    calculation = symmetric - middle
    after = _score(
        _run_group(_Matcher(calculation, labels), np.eye(3), operations, detection, movable=movable, subgroup=subgroup)
    )
    if after != score:
        return None
    return frame, calculation, middle


def oriented_atoms(atoms, positions) -> tuple:
    """``atoms`` at ``positions``, every other field kept."""

    return tuple(
        replace(atom, position=np.array(position, dtype=np.float64)) for atom, position in zip(atoms, positions)
    )


# ------------------------------------------------------------------ outputs


def forces_to_reference(forces, orientation: StandardOrientation | None):
    """A :class:`parsec_python.Forces.ForceResult` of the calculation frame in the reference frame.

    Positions go back by ``x = Q x_calc + d``, every force term by ``Q``; the torque, a sum of ``r x F`` about the
    origin, is taken again from the positions and the symmetrized forces of the reference frame.
    """

    if orientation is None or forces is None:
        return forces
    from ..Forces.symmetry import total_torque

    def vectors(value):
        return None if value is None else orientation.vectors_to_reference(value)

    local = forces.local
    local_fields = {}
    for name in ("ionic", "self_consistency", "core_correction", "gaussian"):
        if hasattr(local, name) and getattr(local, name) is not None:
            local_fields[name] = vectors(getattr(local, name))
    positions = orientation.to_reference(forces.positions)
    symmetrized = vectors(forces.symmetrized)
    return replace(
        forces,
        positions=positions,
        ion_ion=vectors(forces.ion_ion),
        local=replace(local, **local_fields),
        nonlocal_pseudopotential=vectors(forces.nonlocal_pseudopotential),
        unsymmetrized=vectors(forces.unsymmetrized),
        symmetrized=symmetrized,
        net_force=vectors(forces.net_force),
        torque=(
            np.zeros(3)
            if bool(getattr(forces, "periodic", False))
            else total_torque(positions, symmetrized)
        ),
        total=vectors(forces.total),
        point_charge=vectors(getattr(forces, "point_charge", None)),
        hubbard=vectors(getattr(forces, "hubbard", None)),
        dispersion=vectors(getattr(forces, "dispersion", None)),
        spin_orbit=vectors(getattr(forces, "spin_orbit", None)),
    )


def dipole_to_reference(dipole, orientation: StandardOrientation | None):
    """A :class:`parsec_python.Energy.dipole.DipoleMoment` about the origin of the reference frame.

    ``mu = sum_i q_i r_i`` with ``r = Q r_calc + d`` gives ``Q mu_calc + q d`` (``q`` the total charge).
    """

    if orientation is None or dipole is None:
        return dipole
    return replace(
        dipole,
        dipole=orientation.vectors_to_reference(dipole.dipole) + float(dipole.charge) * orientation.centre,
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "OrientationSearch",
    "StandardOrientation",
    "SWITCH",
    "TOLERANCE_SWITCH",
    "dipole_to_reference",
    "find_standard_orientation",
    "forces_to_reference",
    "oriented_atoms",
    "standard_orientation_setting",
    "standard_orientation_tolerance",
]
