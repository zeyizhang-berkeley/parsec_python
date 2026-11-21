"""The levels of a k-point mesh solved as its supercell, k-point by k-point.

A Gamma-centred mesh of one or two points per axis that ``PARSEC_KPOINT_SUPERCELL`` solves as the Gamma point of
its supercell (:mod:`parsec_python.SCF.kpoint_supercell`) gives the union of the levels of the mesh.  Every state of
the supercell belongs to one k-point of the mesh: the translations ``T_a`` of the supercell by one cell along an
axis of two repeats commute with the Hamiltonian and square to the identity, and a Bloch state of the k-point
``k`` has the eigenvalue ``exp(i k_a L_a) = +-1`` (the Bloch phase) under each.  :func:`state_mesh_points` finds
them:

* from the characters of the symmetry sectors whose group holds the translations (the sector of a state fixes
  its phases; the partners of a weighted state are spread over the star of its k-point under the full group,
  as the images that make them);
* or from the orbitals on the full grid: ``<psi|T_a|psi>`` of every state, and within a cluster of degenerate
  levels (whose states may mix k-points) the eigenvalues of ``sum_a 2^a T_a`` in the cluster's span.

:func:`cell_levels` then gives the levels, weights and occupations of every k-point of the k-point route's list
(the irreducible wedge of :func:`parsec_python.Grid.kpoints.kpoint_sampling`), which ``parsec.dat`` and the
result archive of the cell hold (``PARSEC_KPOINT_SUPERCELL_LEVELS=auto``, the default since 2026-11-21; ``off``
deals the sorted levels to the k-points in turn, as before: the union is right, the assignment not).
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import os

import numpy as np

LEVELS_SWITCH = "PARSEC_KPOINT_SUPERCELL_LEVELS"
#: Levels within this many Ry of their neighbour form one cluster (degenerate partners, possibly of several
#: k-points of a star, which a solver may mix).
CLUSTER_TOLERANCE = 1.0e-6
#: A state whose ``|<psi|T_a|psi>|`` is within this of one is that translation's eigenstate.
PHASE_TOLERANCE = 1.0e-6
#: The most bytes of full-grid orbitals formed at a time.
_BLOCK_BYTES = 1 << 30


def supercell_levels_setting() -> str:
    """``PARSEC_KPOINT_SUPERCELL_LEVELS``: ``auto`` (default) or ``off`` (the levels dealt in turn, as before)."""

    value = os.environ.get(LEVELS_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{LEVELS_SWITCH} must be auto or off, not {value!r}")
    return value


def translation_rows(cell_counts, repeats) -> dict[int, np.ndarray]:
    """Per axis of two repeats, the rows ``rows[i]`` of the supercell's grid (row-major, the cell's counts times
    ``repeats``) such that ``psi[rows] = (T_a psi)(r) = psi(r + L_a)``, ``L_a`` the cell's side along ``a``."""

    cell = tuple(int(value) for value in cell_counts)
    repeats = tuple(int(value) for value in repeats)
    if any(value not in (1, 2) for value in repeats):
        raise ValueError("a mesh of one or two points per axis is needed")
    supercounts = tuple(c * m for c, m in zip(cell, repeats))
    index = np.arange(int(np.prod(supercounts)), dtype=np.int64).reshape(supercounts)
    return {
        axis: np.ascontiguousarray(np.roll(index, -cell[axis], axis=axis).reshape(-1))
        for axis in range(3)
        if repeats[axis] == 2
    }


def _mesh_index(bits, axes, repeats) -> int:
    """The index (:func:`numpy.ndindex` order of ``repeats``) of the k-point whose phase is -1 along ``axes[j]``
    where ``bits`` has bit ``j``."""

    index = [0, 0, 0]
    for position, axis in enumerate(axes):
        index[axis] = (int(bits) >> position) & 1
    return int(np.ravel_multi_index(tuple(index), tuple(int(value) for value in repeats)))


def _clusters(eigenvalues: np.ndarray) -> list[np.ndarray]:
    """Runs of the stable sort of ``eigenvalues`` whose neighbours are within :data:`CLUSTER_TOLERANCE`."""

    order = np.argsort(eigenvalues, kind="stable")
    if order.size == 0:
        return []
    values = eigenvalues[order]
    cuts = np.flatnonzero(np.diff(values) > CLUSTER_TOLERANCE) + 1
    return np.split(order, cuts)


def _decode(values: np.ndarray, count: int) -> np.ndarray:
    """The bits of the nearest ``sum_j 2^j s_j`` (``s_j = +-1``) to each value: bit ``j`` set where ``s_j = -1``."""

    patterns = np.arange(1 << count)
    sums = np.array([sum((1 << j) * (-1.0 if (bits >> j) & 1 else 1.0) for j in range(count)) for bits in patterns])
    return patterns[np.argmin(np.abs(np.asarray(values)[:, None] - sums[None, :]), axis=1)]


def _array_module(vectors):
    try:
        import cupy as cp  # noqa: F401

        if isinstance(vectors, cp.ndarray):
            return cp
    except Exception:  # noqa: BLE001 - no CuPy: host arrays only
        pass
    return np


def _batches(clusters: list[np.ndarray], most: int):
    """Consecutive clusters in groups of at most ``most`` states (a larger cluster alone)."""

    batch: list[np.ndarray] = []
    held = 0
    for cluster in clusters:
        if batch and held + cluster.size > most:
            yield batch
            batch, held = [], 0
        batch.append(cluster)
        held += cluster.size
    if batch:
        yield batch


def _dense_bits(blocks, eigenvalues: np.ndarray, rows: dict[int, np.ndarray], most: int = 256) -> np.ndarray:
    """The phase bits of every state from its orbitals: ``blocks(columns)`` gives the full-grid columns (host or
    device) of those states; clusters of degenerate levels are formed together, up to ``most`` states at a time."""

    axes = sorted(rows)
    bits = np.zeros(eigenvalues.size, dtype=np.int64)
    device_rows = {}
    for batch in _batches(_clusters(eigenvalues), most):
        columns = np.concatenate(batch)
        formed = blocks(columns)
        offset = 0
        for cluster in batch:
            vectors = formed[:, offset : offset + cluster.size]
            offset += cluster.size
            _cluster_bits(vectors, cluster, bits, rows, axes, device_rows)
        del formed
    return bits


def _cluster_bits(vectors, cluster, bits, rows, axes, device_rows) -> None:
    """The phase bits of the states ``cluster`` (columns ``vectors``, host or device) into ``bits``."""

    xp = _array_module(vectors)
    if xp is not np and not device_rows:
        device_rows.update({axis: xp.asarray(rows[axis]) for axis in axes})
    index = device_rows if xp is not np else rows
    gram = np.asarray(_host(vectors.T @ vectors), dtype=np.float64)
    moved = [np.asarray(_host(vectors.T @ vectors[index[axis], :]), dtype=np.float64) for axis in axes]
    norms = np.sqrt(np.maximum(np.diag(gram), 1.0e-300))
    diagonal = np.array([np.diag(matrix) / norms**2 for matrix in moved])  # (axes, states)
    if np.all(np.abs(np.abs(diagonal) - 1.0) <= PHASE_TOLERANCE):
        found = np.zeros(cluster.size, dtype=np.int64)
        for position in range(len(axes)):
            found |= (diagonal[position] < 0.0).astype(np.int64) << position
        bits[cluster] = found
        return
    # a cluster whose states mix k-points: the eigenvalues of sum_j 2^j T_j in its span (the generalized
    # problem with its Gram matrix), each a +-1 pattern of the translations
    import scipy.linalg

    combined = sum((1 << position) * matrix for position, matrix in enumerate(moved))
    combined = 0.5 * (combined + combined.T)
    values = scipy.linalg.eigh(combined, 0.5 * (gram + gram.T), eigvals_only=True)
    bits[cluster] = _decode(values, len(axes))


def _host(values):
    return values.get() if hasattr(values, "get") and not isinstance(values, np.ndarray) else values


def _character_bits(orbitals, rows: dict[int, np.ndarray]) -> np.ndarray | None:
    """Per sector of ``orbitals`` (:class:`CuPySymmetryOrbitals`), the phase bits of its states, where the group
    of the sectors holds every translation of ``rows``; ``None`` otherwise."""

    full_to_wedge = np.asarray(_host(orbitals.full_to_wedge), dtype=np.int64).reshape(-1)
    axes = sorted(rows)
    if any(full_to_wedge.size != rows[axis].size for axis in axes):
        return None
    if any(not np.array_equal(full_to_wedge[rows[axis]], full_to_wedge) for axis in axes):
        return None
    count = int(orbitals.phases.shape[0])
    bits = np.zeros(count, dtype=np.int64)
    for sector in range(count):
        phases = np.asarray(_host(orbitals.phases[sector]), dtype=np.float64).reshape(-1)
        for position, axis in enumerate(axes):
            product = phases[rows[axis]] * phases
            kept = product != 0.0
            if not np.any(kept):
                continue  # a sector of no orbit: no state
            signs = np.sign(product[kept])
            if not (np.all(signs == signs[0]) and np.allclose(np.abs(phases[rows[axis]][kept]), np.abs(phases[kept]))):
                return None
            if signs[0] < 0.0:
                bits[sector] |= 1 << position
    return bits


def _star(bits: int, axes, operations) -> list[int]:
    """The phase bits of the k-points of the star of ``bits`` under the rotations ``operations`` (signed
    permutations of the axes), the k-point itself first."""

    axes = list(axes)
    star = [int(bits)]
    for matrix in np.asarray(operations, dtype=np.int64).reshape(-1, 3, 3):
        image, on_mesh = 0, True
        for target in range(3):
            source = int(np.flatnonzero(matrix[target])[0])  # (R k)_target = +-k_source
            phase = source in axes and (int(bits) >> axes.index(source)) & 1
            if not phase:
                continue
            if target not in axes:
                on_mesh = False  # an image off the mesh: no symmetry of the supercell does that
                break
            image |= 1 << axes.index(target)
        if on_mesh and image not in star:
            star.append(image)
    return star


def _partner_bits(solved_bits, weights, eigenvalues, axes, operations) -> np.ndarray:
    """The bits of every listed state (the partners of a weighted solved state after it, as the result lists
    them): a solved state of weight 1 keeps its own; the partners of the weighted states of one level and one star
    are spread evenly over the star (the images under the full group that make them), the rest in turn."""

    weights = np.asarray(weights, dtype=np.int64)
    solved_bits = np.asarray(solved_bits, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(weights)[:-1]])
    listed = np.zeros(int(weights.sum()), dtype=np.int64)
    for state in range(weights.size):
        listed[starts[state] : starts[state] + weights[state]] = solved_bits[state]
    weighted = np.flatnonzero(weights > 1)
    if weighted.size == 0:
        return listed
    solved_values = np.asarray(eigenvalues, dtype=np.float64)[starts]
    for cluster in _clusters(solved_values[weighted]):
        members = weighted[cluster]
        by_star: dict[tuple, list[int]] = {}
        for state in members:
            star = _star(int(solved_bits[state]), axes, operations)
            by_star.setdefault(tuple(sorted(star)), []).append(int(state))
        for star, states in by_star.items():
            total = int(weights[states].sum())
            if total % len(star) == 0:
                share = [point for point in star for _ in range(total // len(star))]
            else:
                share = []
                for state in states:
                    own = _star(int(solved_bits[state]), axes, operations)
                    share.extend(own[position % len(own)] for position in range(int(weights[state])))
            columns = np.concatenate([np.arange(starts[s], starts[s] + weights[s]) for s in states])
            listed[columns] = np.asarray(share, dtype=np.int64)
    return listed


def _translation_maps(rows: dict[int, np.ndarray], axes) -> list[np.ndarray]:
    """The row maps of every element of the group of the translations of ``rows``: element ``b`` (bit ``j`` for
    ``T_{axes[j]}``) gives ``(T_b psi)[i] = psi[maps[b][i]]``."""

    size = int(rows[axes[0]].size)
    maps = []
    for bits in range(1 << len(axes)):
        index = np.arange(size, dtype=np.int64)
        for position, axis in enumerate(axes):
            if (bits >> position) & 1:
                index = index[rows[axis]]
        maps.append(index)
    return maps


def _weighted_bits(orbitals, eigenvalues, weights, rows, axes, operations, most: int) -> np.ndarray:
    """The bits of every listed state of weighted sectors whose group lacks the translations of the cell.

    Each solved state (formed on the full grid without its partners) is a sum over the k-points of one star (those
    of its level): its weights on the k-points, ``p_k = <psi|P_k|psi>`` with the projector ``P_k`` of the
    translation group, give that star under the full group (``operations``); the states of a level and a star,
    with their partners, are then spread evenly over the star, as their images are (in turn where the count does
    not divide)."""

    maps = _translation_maps(rows, axes)
    count = len(maps)
    characters = np.array(
        [[-1.0 if bin(point & element).count("1") % 2 else 1.0 for element in range(count)] for point in range(count)]
    )
    starts = np.concatenate([[0], np.cumsum(weights)[:-1]]).astype(np.int64)
    solved_values = np.asarray(eigenvalues, dtype=np.float64)[starts]
    stars: list[tuple[int, ...]] = [()] * weights.size
    device_maps = None
    order = np.argsort(solved_values, kind="stable")
    for begin in range(0, order.size, most):
        batch = order[begin : begin + most]
        vectors = _sector_columns(orbitals, batch)
        xp = _array_module(vectors)
        if device_maps is None:
            device_maps = [xp.asarray(item) for item in maps] if xp is not np else maps
        norms = np.asarray(_host((vectors * vectors).sum(axis=0)), dtype=np.float64)
        expectations = np.array(
            [np.asarray(_host((vectors * vectors[item, :]).sum(axis=0)), dtype=np.float64) for item in device_maps]
        ) / np.maximum(norms, 1.0e-300)[None, :]
        weights_k = characters @ expectations / count  # (points, states)
        for column, state in enumerate(batch):
            support = [int(point) for point in np.flatnonzero(weights_k[:, column] > 0.05)]
            star: set[int] = set()
            for point in support or [int(np.argmax(weights_k[:, column]))]:
                star.update(_star(point, axes, operations))
            stars[int(state)] = tuple(sorted(star))
        del vectors
    listed = np.zeros(int(np.sum(weights)), dtype=np.int64)
    for cluster in _clusters(solved_values):
        by_star: dict[tuple, list[int]] = {}
        for state in cluster:
            by_star.setdefault(stars[int(state)], []).append(int(state))
        for star, states in by_star.items():
            total = int(np.sum(weights[states]))
            if total % len(star) == 0:
                share = [point for point in star for _ in range(total // len(star))]
            else:
                share = [star[position % len(star)] for position in range(total)]
            columns = np.concatenate([np.arange(starts[s], starts[s] + weights[s]) for s in states])
            listed[columns] = np.asarray(share, dtype=np.int64)
    return listed


def _sector_columns(orbitals, solved):
    """The full-grid columns (device) of the solved states ``solved`` of :class:`CuPySymmetryOrbitals`."""

    solved = np.asarray(solved, dtype=np.int64)
    changes = {
        "representations": np.asarray(orbitals.representations)[solved],
        "state_weights": None,
    }
    if getattr(orbitals, "representation_columns", None) is not None:
        changes["representation_columns"] = np.asarray(orbitals.representation_columns)[solved]
    if getattr(orbitals, "scaled_wedge_vectors", None) is not None:
        changes["scaled_wedge_vectors"] = orbitals.scaled_wedge_vectors[:, solved]
    return replace(orbitals, **changes).to_full_device()


def state_mesh_points(result, cell_counts, repeats) -> tuple[np.ndarray | None, str | None]:
    """``(points, note)``: the index of the k-point of the mesh ``repeats`` (:func:`numpy.ndindex` order) of every
    listed state of the supercell's ``result`` (its ``eigenvalues`` order), or ``None`` with the reason where
    its orbitals do not tell (none kept, orbitals on other MPI ranks of a group without the translations)."""

    eigenvalues = np.asarray(_host(result.eigenvalues), dtype=np.float64).reshape(-1)
    repeats = tuple(int(value) for value in repeats)
    rows = translation_rows(cell_counts, repeats)
    axes = sorted(rows)
    if not axes:
        return np.zeros(eigenvalues.size, dtype=np.int64), None
    orbitals = getattr(result, "wavefunctions", None)
    size = int(rows[axes[0]].size)
    # full-grid columns formed at a time (a cluster of degenerate levels larger than that alone)
    chunk = max(1, min(256, _BLOCK_BYTES // (8 * size)))

    def to_points(bits):
        return np.array([_mesh_index(value, axes, repeats) for value in np.asarray(bits)], dtype=np.int64)

    if orbitals is None:
        return None, "the result keeps no orbitals"
    if hasattr(orbitals, "phases") and hasattr(orbitals, "full_to_wedge"):
        representations = np.asarray(orbitals.representations, dtype=np.int64)
        weights = getattr(orbitals, "state_weights", None)
        weights = np.ones(representations.size, dtype=np.int64) if weights is None else np.asarray(weights, np.int64)
        if int(weights.sum()) != eigenvalues.size:
            return None, "the orbitals do not list the states of the levels"
        operations = getattr(getattr(orbitals, "point_group", None), "full_operations", None)
        if operations is None:
            operations = np.eye(3, dtype=np.int64)[None]
        sector_bits = _character_bits(orbitals, rows)
        if sector_bits is not None:
            solved_bits = sector_bits[representations]
            return to_points(_partner_bits(solved_bits, weights, eigenvalues, axes, operations)), None
        if getattr(orbitals, "remote_sectors", False):
            return None, "sector orbitals on other MPI ranks whose group lacks the translations of the cell"
        try:
            if np.any(weights > 1):
                # weighted sectors without the translations: the k-points of the star of every solved state
                return to_points(_weighted_bits(orbitals, eigenvalues, weights, rows, axes, operations, chunk)), None
            solved_bits = _dense_bits(lambda columns: _sector_columns(orbitals, columns), eigenvalues, rows, most=chunk)
        except (MemoryError, RuntimeError) as error:
            return None, f"the orbitals could not be formed on the full grid ({error})"
        return to_points(solved_bits), None
    shape = getattr(orbitals, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != size or int(shape[1]) != eigenvalues.size:
        return None, "the orbitals are not full-grid columns of the listed states"

    def dense(columns):
        return orbitals[:, np.asarray(columns, dtype=np.int64)]

    return to_points(_dense_bits(dense, eigenvalues, rows, most=chunk)), None


@dataclass(frozen=True)
class CellLevels:
    """The levels of the k-points of the cell's list: ``eigenvalues`` and ``occupations`` k-major (``per_k`` per
    k-point, each block sorted), ``k_points`` (Cartesian, 1/bohr) and ``weights`` those of the k-point route,
    ``note`` what the files say about them (``None``: nothing)."""

    k_points: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray
    per_k: int
    note: str | None = None
    #: the states of a k-point of the run (the supercell's over the k-points of the mesh)
    states: int = 0


def _wedge_points(cell_problem, repeats):
    """``(k_points, weights, mesh index of each)`` of the k-point route's list of the cell, or of the whole mesh
    with equal weights where that list is not on the mesh."""

    from parsec_python.Grid.cell import reciprocal_lattice
    from parsec_python.Grid.kpoints import kpoint_sampling

    from .kpoint_supercell import mesh_kpoints

    repeats = tuple(int(value) for value in repeats)
    mesh = mesh_kpoints(cell_problem, repeats)
    count = int(np.prod(repeats))
    whole = (mesh, np.full(count, 1.0 / count), np.arange(count, dtype=np.int64))
    try:
        sampling = kpoint_sampling(cell_problem)
    except Exception:  # noqa: BLE001 - the whole mesh then
        return whole
    if sampling is None:
        return whole
    k_points, weights = (np.asarray(value, dtype=np.float64) for value in sampling)
    k_points = k_points.reshape(-1, 3)
    fractions = k_points @ np.linalg.inv(reciprocal_lattice(np.asarray(cell_problem.periodic_cell.lattice_vectors)))
    scaled = fractions * np.asarray(repeats, dtype=np.float64)[None, :]
    if not np.allclose(scaled, np.round(scaled), atol=1.0e-6):
        return whole
    indices = np.mod(np.round(scaled).astype(np.int64), np.asarray(repeats, dtype=np.int64)[None, :])
    points = np.array([np.ravel_multi_index(tuple(row), repeats) for row in indices], dtype=np.int64)
    if abs(float(weights.sum()) - 1.0) > 1.0e-8:
        return whole
    return k_points, weights, points


def dealt_levels(eigenvalues, count: int) -> np.ndarray:
    """The order that deals the levels of the supercell to the ``count`` k-points of the mesh in turn from the lowest
    (the union is the mesh's, the assignment to each k-point not): ``PARSEC_KPOINT_SUPERCELL_LEVELS=off``."""

    eigenvalues = np.asarray(eigenvalues)
    order = np.argsort(eigenvalues, kind="stable")
    per_k = eigenvalues.size // count
    return np.concatenate([order[k::count][:per_k] for k in range(count)])


#: The levels of the last result (a weak reference to it, the identity of its eigenvalues, the levels): the
#: ``parsec.dat`` and the archive of one run read them once.
_CACHE: dict[int, tuple[object, int, CellLevels]] = {}


def _cached(result, eigenvalue_source) -> CellLevels | None:
    entry = _CACHE.get(id(result))
    if entry is None:
        return None
    reference, source, found = entry
    return found if reference() is result and source == id(eigenvalue_source) else None


def _remember(result, eigenvalue_source, found: CellLevels) -> None:
    import weakref

    _CACHE.clear()
    try:
        _CACHE[id(result)] = (weakref.ref(result), id(eigenvalue_source), found)
    except TypeError:  # an object without weak references: computed again when asked again
        pass


def cell_levels(result, cell_problem, repeats) -> CellLevels:
    """The levels of the k-points of the cell of ``cell_problem`` from the supercell's ``result``.

    Under ``PARSEC_KPOINT_SUPERCELL_LEVELS=auto`` every state goes to the k-point its Bloch phases give
    (:func:`state_mesh_points`); each k-point of the k-point route's list (its irreducible wedge, with its
    weights) keeps its lowest ``per_k`` levels, ``per_k`` the fewest any k-point of the mesh holds, with their
    occupations (electrons per state, as the supercell's: the mesh's equal weights and one Fermi level).  Where
    the orbitals do not tell, and under ``off``, the levels are dealt to the k-points of the whole mesh in turn
    (the union is the mesh's), with a note where the orbitals did not tell (none under ``off``, the output of
    before).  One computation per result.
    """

    from parsec_python.Grid.pbc import build_periodic_grid

    from .kpoint_supercell import mesh_kpoints

    eigenvalue_source = getattr(result, "eigenvalues", None)
    cached = _cached(result, eigenvalue_source)
    if cached is not None:
        return cached
    repeats = tuple(int(value) for value in repeats)
    count = int(np.prod(repeats))
    eigenvalues = np.asarray(_host(result.eigenvalues), dtype=np.float64).reshape(-1)
    occupations = np.asarray(_host(result.occupations), dtype=np.float64).reshape(-1)
    points, note = None, None
    if supercell_levels_setting() == "auto":
        cell_grid = build_periodic_grid(cell_problem.periodic_cell, cell_problem.grid)
        points, reason = state_mesh_points(result, cell_grid.points_per_axis, repeats)
        if points is None:
            note = (
                "the levels of parsec.dat and of the result archive are those of the supercell dealt to the k-points "
                f"of the mesh in turn (their union is right, not the k-point of each): {reason}"
            )
    # (off: no note, the output of before)
    if points is None:
        dealt = dealt_levels(eigenvalues, count)
        levels = CellLevels(
            k_points=mesh_kpoints(cell_problem, repeats),
            weights=np.full(count, 1.0 / count),
            eigenvalues=eigenvalues[dealt],
            occupations=occupations[dealt],
            per_k=eigenvalues.size // count,
            note=note,
            states=eigenvalues.size // count,
        )
    else:
        k_points, weights, wedge = _wedge_points(cell_problem, repeats)
        held = np.bincount(points, minlength=count)
        per_k = int(held.min())
        blocks_values, blocks_occupations = [], []
        for point in wedge:
            members = np.flatnonzero(points == point)
            members = members[np.argsort(eigenvalues[members], kind="stable")][:per_k]
            blocks_values.append(eigenvalues[members])
            blocks_occupations.append(occupations[members])
        levels = CellLevels(
            k_points=k_points,
            weights=weights,
            eigenvalues=np.concatenate(blocks_values) if blocks_values else np.zeros(0),
            occupations=np.concatenate(blocks_occupations) if blocks_occupations else np.zeros(0),
            per_k=per_k,
            states=eigenvalues.size // count,
            note=(
                None
                if per_k >= eigenvalues.size // count
                else (
                    f"each k-point keeps the {per_k} lowest levels its Bloch phases give it, of the "
                    f"{eigenvalues.size // count} states of a k-point (the supercell's {eigenvalues.size} levels "
                    f"hold {int(held.min())} to {int(held.max())} per k-point); a Fix_Number_Eigenvalues restart from "
                    f"this file needs States_Num {per_k} or fewer"
                )
            ),
        )
    _remember(result, eigenvalue_source, levels)
    return levels


__all__ = [
    "CLUSTER_TOLERANCE",
    "CellLevels",
    "LEVELS_SWITCH",
    "cell_levels",
    "dealt_levels",
    "state_mesh_points",
    "supercell_levels_setting",
    "translation_rows",
]
