"""Brillouin-zone k-point sampling for a periodic cell.

Three sources are supported:

* Fortran PARSEC's grid ``k = ((i-1) + s_i)/n_i`` reduced to the irreducible
  wedge by the space group and time reversal
  (:func:`irreducible_monkhorst_pack`, ``irrbz.f90``): unequal weights
  ``count/N``.  An input that gives ``Monkhorst_Pack_Shift`` takes its
  shift; one without the block takes ``s = 0`` (the Gamma-centred mesh), as
  PARSEC 1.5 does: ``usrinputfile.F90:976-994`` then leaves ``mpshift`` at
  its static zero and ``init_var.F90:281-289`` calls ``irrbz`` for every
  Monkhorst-Pack input (:func:`unshifted_mesh_setting`, since 2026-11-18;
  a mesh of one point stays the Gamma point of the input);
* this port's former uniform, unreduced Monkhorst-Pack grid
  (:func:`monkhorst_pack_grid`, fractions ``(2i-n-1)/(2n)``, equal weights),
  which an input without a ``Monkhorst_Pack_Shift`` block takes with
  ``PARSEC_MP_UNSHIFTED_MESH=port`` (every such input before 2026-11-18);
* a user-supplied explicit list (:func:`explicit_kpoints`,
  :func:`read_kpoint_file`, Fortran's ``Kpoint_Method: manual``), whose
  weights are normalized to one as ``init_var.F90`` does.

For odd ``n`` the former grid equals Fortran's with ``s = 0`` modulo
reciprocal lattice vectors, for even ``n`` Fortran's with ``s = 0.5``.  The
finite-difference Bloch Hamiltonian is not exactly periodic in ``k``, so
equivalent points give slightly different eigenvalues; the Fortran path keeps
Fortran's representatives (folded into ``(-1/2, 1/2]``,
:func:`fold_to_centered_zone`).

Equal weights keep the scalar-``degeneracy`` Fermi-level pooling of the SCF
drivers (``degeneracy = 2*weight``); unequal weights use the per-state
weights of :func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`
(Fortran ``flevel``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..models import PeriodicCell
from .cell import is_diagonal_lattice, reciprocal_lattice
from .space_group import fortran_anint, fortran_reciprocal_lattice


def monkhorst_pack_grid(
    cell: PeriodicCell,
    dimensions: tuple[int, int, int],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_points, weights)`` for a uniform Monkhorst-Pack grid.

    ``dimensions=(n1,n2,n3)`` are the number of divisions along each
    reciprocal lattice vector.  For an orthorhombic cell with lattice
    vectors ``diag(Lx,Ly,Lz)``, the reciprocal lattice vectors are
    ``b_a = (2*pi/L_a)*e_a``, and the unshifted grid points are

    ``k_(i1,i2,i3) = sum_a [(2*i_a-n_a-1)/(2*n_a)] * b_a``,
    ``i_a = 1,...,n_a``,

    the standard Monkhorst-Pack construction, optionally displaced by
    ``shift`` (given in the same Cartesian reciprocal units as the returned
    k-points, e.g. a half-grid-step shift to avoid sampling exactly at
    Gamma).  Every returned weight is exactly ``1/(n1*n2*n3)``: this is the
    full uniform grid, not reduced to the irreducible wedge.

    ``k_points`` has shape ``(n1*n2*n3, 3)``, ``weights`` has shape
    ``(n1*n2*n3,)`` and sums to 1.

    A general (non-orthorhombic) cell uses the same fractions of its
    reciprocal lattice vectors, ``k = sum_a f_a b_a`` with
    ``b = 2*pi*inv(A)^T`` (rows), in Cartesian 1/bohr.  (Fortran's set is
    ``f = (i-1+s)/n`` in ``[0, 1)``, see report section 1.3: equal modulo
    ``G`` to this one for odd ``n`` without a shift.)
    """
    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    if not is_diagonal_lattice(cell.lattice_vectors):
        shift = np.asarray(shift, dtype=float)
        if shift.shape != (3,):
            raise ValueError("shift must have shape (3,)")
        fractional_axes = [
            (2.0 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in dims
        ]
        grids = np.meshgrid(*fractional_axes, indexing="ij")
        fractional = np.column_stack([axis.reshape(-1) for axis in grids])
        k_points = fractional @ reciprocal_lattice(cell.lattice_vectors) + shift[None, :]
        n_total = int(np.prod(dims))
        return k_points, np.full(n_total, 1.0 / n_total, dtype=float)
    side_lengths = np.diag(cell.lattice_vectors)
    reciprocal = 2.0 * np.pi / side_lengths
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (3,):
        raise ValueError("shift must have shape (3,)")

    fractional_axes = [
        (2.0 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in dims
    ]
    grids = np.meshgrid(*fractional_axes, indexing="ij")
    fractional = np.column_stack([axis.reshape(-1) for axis in grids])
    if reciprocal.ndim == 2:
        k_points = fractional @ reciprocal + shift[None, :]
    else:
        k_points = fractional * reciprocal[None, :] + shift[None, :]
    n_total = int(np.prod(dims))
    weights = np.full(n_total, 1.0 / n_total, dtype=float)
    return k_points, weights


KPOINT_UNITS = ("cartesian_inverse_bohr", "reciprocal_lattice_vectors")


def explicit_kpoints(
    rows,
    *,
    unit: str = "cartesian_inverse_bohr",
    cell: PeriodicCell | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a user k-point list to ``(k_points, weights)``.

    ``rows`` is ``(n, 3)`` (equal weights are assumed) or ``(n, 4)`` with the
    sampling weight in the last column, the Fortran ``Kpoint_List`` row
    ``kx ky kz weight``.  ``unit`` is either ``"cartesian_inverse_bohr"``
    (the Fortran default: Cartesian components in 1/bohr) or
    ``"reciprocal_lattice_vectors"`` (fractional coordinates along the
    reciprocal lattice vectors ``b_i``, so ``k = sum_i f_i b_i`` with
    ``b = 2*pi*inv(A).T`` for lattice vectors ``A`` in the rows; this needs
    ``cell``).  The returned k-points are always Cartesian 1/bohr; weights
    are normalized to sum to 1 and must be non-negative.
    """

    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] not in (3, 4):
        raise ValueError(
            "k-points must be an (n, 3) array or an (n, 4) array with a weight column"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("k-points and weights must be finite")
    coordinates = array[:, :3]
    if array.shape[1] == 4:
        weights = array[:, 3].copy()
    else:
        weights = np.ones(array.shape[0])
    if np.any(weights < 0.0):
        raise ValueError("k-point sampling weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("k-point sampling weights must not all be zero")
    weights = weights / total

    normalized_unit = unit.strip().lower()
    if normalized_unit not in KPOINT_UNITS:
        raise ValueError(
            f"Kpoint_Unit must be one of {', '.join(KPOINT_UNITS)}; got {unit!r}"
        )
    if normalized_unit == "reciprocal_lattice_vectors":
        if cell is None:
            raise ValueError("reciprocal_lattice_vectors units need the periodic cell")
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(cell.lattice_vectors)).T
        coordinates = coordinates @ reciprocal
    return np.ascontiguousarray(coordinates), weights


def read_kpoint_file(path) -> np.ndarray:
    """Read a k-point file: one ``kx ky kz [weight]`` row per line.

    Blank lines and anything after ``#`` or ``!`` are ignored.  All rows must
    have the same number of columns (three, or four with a weight).  Returns
    the raw ``(n, 3)`` or ``(n, 4)`` array; apply units with
    :func:`explicit_kpoints`.
    """

    from pathlib import Path

    source = Path(path)
    rows: list[list[float]] = []
    for number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw
        for marker in ("#", "!"):
            position = line.find(marker)
            if position >= 0:
                line = line[:position]
        fields = line.split()
        if not fields:
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{source}: line {number}: non-numeric entry") from error
        if len(values) not in (3, 4):
            raise ValueError(
                f"{source}: line {number}: expected 'kx ky kz [weight]', got {len(values)} numbers"
            )
        rows.append(values)
    if not rows:
        raise ValueError(f"{source}: no k-points found")
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{source}: every row must have the same number of columns")
    return np.asarray(rows, dtype=float)


def band_path_kpoints(
    starts: np.ndarray,
    ends: np.ndarray,
    points_in_shortest_segment: int,
    *,
    rounding: str = "truncate",
) -> list[np.ndarray]:
    """K-points of each band-path segment, as in Fortran ``calc_bands``.

    ``starts``/``ends`` are the Cartesian (1/bohr) end points of every
    segment.  The shortest segment gets ``points_in_shortest_segment`` points
    and the others proportionally more, so the point density along the path is
    constant.  ``rounding="truncate"`` (the default, this port's rule and that
    of the previous Fortran tree): ``nkpt = int(length / min_length * points)``,
    the Fortran integer assignment.  ``rounding="nint"`` (PARSEC 1.5,
    ``utilities/calc_bands.f90:152-158``, which the parser takes where the
    input writes ``Bandstruc_Points``): ``nint(length / min_length * points)``.
    A segment does not include its end point -- that is the first point of the
    next segment -- except the last segment, which gets one extra point to
    close the path (``int(...) + 1``, or 1.5's ``nint(... + 1)``).
    """

    if rounding not in ("truncate", "nint"):
        raise ValueError("rounding must be truncate or nint")
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("segment end points must be two (n, 3) arrays")
    if starts.shape[0] < 1:
        raise ValueError("a band path needs at least one segment")
    if points_in_shortest_segment < 1:
        raise ValueError("bandstruc_points must be at least 1")
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("band-path segments must have non-zero length")
    shortest = float(lengths.min())
    segments = []
    last = starts.shape[0] - 1
    for index, (start, end, length) in enumerate(zip(starts, ends, lengths)):
        if rounding == "truncate":
            count = int(length / shortest * points_in_shortest_segment)
            if index == last:
                count += 1
        else:
            scaled = length / shortest * points_in_shortest_segment
            if index == last:
                scaled += 1.0
            # Fortran nint: halves away from zero (the values are positive).
            count = int(np.floor(scaled + 0.5))
        if index == last:
            fractions = np.arange(count) / (count - 1) if count > 1 else np.zeros(1)
        else:
            fractions = np.arange(count) / count
        segments.append(start + fractions[:, None] * (end - start))
    return segments


IBZ_TOLERANCE = 1.0e-8
"""``tol_bz`` of ``irrbz.f90`` (and of ``translate01``)."""


def translate01(vector) -> list[float]:
    """Fortran ``translate01`` (``irrbz.f90:213-254``), component by component.

    ``q = mod(k + 1/2, 1) - 1/2``, set to zero below ``1e-8`` in magnitude and
    raised by one when negative.  The result lies in ``[0, 1)`` for
    ``k > -1/2`` (every input of ``irrbz``); the operations are those of the
    Fortran, so the bits agree.
    """

    result = []
    for value in vector:
        shifted = float(value) + 0.5
        shifted = math.fmod(shifted, 1.0)
        shifted = shifted - 0.5
        if abs(shifted) < IBZ_TOLERANCE:
            shifted = 0.0
        if shifted < 0.0:
            shifted = shifted + 1.0
        result.append(shifted)
    return result


def fold_to_centered_zone(fractions) -> np.ndarray:
    """Reciprocal-lattice coordinates folded component by component into ``(-1/2, 1/2]``.

    ``f - ceil(f - 1/2)``: the representative of ``f`` modulo the reciprocal lattice with the smallest
    components (the shortest one of an orthorhombic cell), PARSEC 1.5's intent in ``irrbz.f90:132-157``, whose
    swap never runs (``norm2`` carries ``+ eight``).  Exactly ``1/2`` stays ``1/2`` (``-1/2`` becomes ``1/2``): a
    component in ``[0, 1/2]`` keeps its bits, so a grid whose points all lie there is unchanged, and folding a
    folded set changes nothing.  The finite-difference Bloch operator ``-nabla^2 - 2i k.grad + k^2`` is not
    invariant under ``k -> k + G``; with the folded points a run without symmetry takes the k-points a symmetric
    run reaches by rotating its representatives.
    """

    values = np.asarray(fractions, dtype=np.float64)
    return values - np.ceil(values - 0.5)


#: Two representatives of a k-point modulo the reciprocal lattice are equally short where their squared lengths
#: differ by less than this times the largest ``|b_i|^2`` (:func:`fold_to_shortest`).
SHORTEST_FOLD_TOLERANCE = 1.0e-10


def fold_to_shortest(fractions, reciprocal, *, tolerance: float = SHORTEST_FOLD_TOLERANCE) -> np.ndarray:
    """Reciprocal-lattice coordinates moved to their shortest representatives modulo the reciprocal lattice.

    ``fractions`` are rows ``f`` (or one row) of ``k = f B``, ``reciprocal`` the rows ``b_i`` of ``B``.  Each point is
    first folded into ``(-1/2, 1/2]`` (:func:`fold_to_centered_zone`) and then moved by the reciprocal vector ``G``
    that makes ``|k - G|`` smallest, the point of the first Brillouin zone (the Wigner-Seitz cell of the reciprocal
    lattice).  The folded point is kept where it is as short as the shortest within ``tolerance`` (relative to the
    largest ``|b_i|^2``), so on a lattice of orthogonal vectors, where every folded point is a shortest one, nothing
    changes bit for bit; among other equally short points the first of the candidates in the order of
    ``(|n|_1, n)`` is taken.  ``|k - G| <= |k|`` bounds every component of ``G = n B``:
    ``|f_i - n_i| <= |k| |a_i| / 2 pi``, so the candidates are searched exhaustively on any cell.

    Why: the finite-difference Bloch operator of the u gauge, ``-nabla^2 - 2i k.grad + k^2`` on the periodic factor
    ``u_k``, is not invariant under ``k -> k + G`` (its error grows with the plane waves of ``u_k``, whose leading one
    is ``e^{-iG.r}`` where ``k`` is moved by ``G`` out of the first zone).  On a lattice of orthogonal vectors the
    folded points are already the shortest; on an oblique one (hexagonal, the primitive fcc and bcc cells) a point
    with components ``1/2`` and ``1/4``, for instance, can lie outside the first zone.
    """

    values = fold_to_centered_zone(fractions)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    if values.shape[0] == 0:
        return values
    basis = np.asarray(reciprocal, dtype=np.float64).reshape(3, 3)
    metric = basis @ basis.T
    scale = float(np.max(np.diag(metric)))

    def squared(points):
        return np.einsum("...i,ij,...j->...", points, metric, points)

    lengths = squared(values)
    radius = math.sqrt(max(float(np.max(lengths)), 0.0))
    # |a_i| / 2 pi: the columns of B^-1
    directions = np.linalg.norm(np.linalg.inv(basis), axis=0)
    reach = [int(math.floor(0.5 + radius * float(value) + 1.0e-9)) for value in directions]
    import itertools

    candidates = [
        n
        for n in itertools.product(*(range(-r, r + 1) for r in reach))
        if any(n)
    ]
    candidates.sort(key=lambda n: (sum(abs(v) for v in n), n))
    best = lengths.copy()
    moves = np.zeros_like(values)
    for n in candidates:
        shift = np.asarray(n, dtype=np.float64)
        trial = squared(values - shift[None, :])
        better = trial < best - tolerance * scale
        if np.any(better):
            best = np.where(better, trial, best)
            moves[better] = shift
    moved = np.any(moves != 0.0, axis=1)
    result = values.copy()
    result[moved] = values[moved] - moves[moved]
    return result[0] if single else result


def oblique_reciprocal_lattice(reciprocal, *, tolerance: float = 1.0e-12) -> bool:
    """Whether the reciprocal vectors (rows of ``reciprocal``) are not mutually orthogonal (relative to the largest
    ``|b_i|^2``): a hexagonal, monoclinic or triclinic cell, the primitive fcc and bcc cells.  On a lattice of
    orthogonal vectors every point folded into ``(-1/2, 1/2]`` is a shortest one (:func:`fold_to_shortest`)."""

    basis = np.asarray(reciprocal, dtype=np.float64).reshape(3, 3)
    metric = basis @ basis.T
    scale = float(np.max(np.abs(np.diag(metric))))
    return bool(np.max(np.abs(metric - np.diag(np.diag(metric)))) > tolerance * scale)


def shortest_images(fractions, reciprocal, *, tolerance: float = SHORTEST_FOLD_TOLERANCE) -> list[np.ndarray]:
    """Every shortest representative of each point modulo the reciprocal lattice, the ties on the zone boundary
    included.

    ``fractions`` are rows ``f`` of ``k = f B`` (``reciprocal``: the rows ``b_i`` of ``B``).  Entry ``i`` of the list
    is an ``(m, 3)`` array of the points ``f_i - n`` (``f_i`` folded into ``(-1/2, 1/2]`` first) whose squared lengths
    are within ``tolerance`` (relative to the largest ``|b_i|^2``) of the shortest: one point inside the first
    Brillouin zone, two or more on its boundary (the faces, edges and corners of the Wigner-Seitz cell, such as the
    fcc ``X``, ``L`` and ``W`` points or the hexagonal ``M`` and ``K`` points).  The first is that of
    :func:`fold_to_shortest`, the others follow in the order of its candidates.  The finite-difference Bloch operator
    is not invariant under ``k -> k + G``, so equally short images have levels that differ at the order of the
    discretization unless an operation that keeps the grid's finite-difference operators relates them; a mesh point
    with ``m`` of them is computed as all of them, each with ``1/m`` of its weight (the choice that does not depend on
    an order of the candidates and that every operation of the lattice maps onto itself).

    On a lattice of mutually orthogonal reciprocal vectors (:func:`oblique_reciprocal_lattice` false) the folded point
    alone is returned, as :func:`fold_to_shortest` gives it: a component ``1/2`` keeps its tie with ``-1/2`` there
    (the points of these lattices before 2026-11-21, bit for bit).
    """

    values = np.atleast_2d(fold_to_centered_zone(fractions))
    if values.shape[0] == 0:
        return []
    basis = np.asarray(reciprocal, dtype=np.float64).reshape(3, 3)
    if not oblique_reciprocal_lattice(basis):
        return [row[None, :].copy() for row in values]
    first = np.atleast_2d(fold_to_shortest(values, basis, tolerance=tolerance))
    metric = basis @ basis.T
    scale = float(np.max(np.diag(metric)))
    lengths = np.einsum("...i,ij,...j->...", values, metric, values)
    radius = math.sqrt(max(float(np.max(lengths)), 0.0))
    directions = np.linalg.norm(np.linalg.inv(basis), axis=0)
    reach = [int(math.floor(0.5 + radius * float(value) + 1.0e-9)) for value in directions]
    import itertools

    candidates = sorted(
        itertools.product(*(range(-r, r + 1) for r in reach)), key=lambda n: (sum(abs(v) for v in n), n)
    )
    shifts = np.asarray(candidates, dtype=np.float64)
    # (points, candidates): the squared lengths of f - n
    moved = values[:, None, :] - shifts[None, :, :]
    trial = np.einsum("pci,ij,pcj->pc", moved, metric, moved)
    shortest = trial <= np.min(trial, axis=1, keepdims=True) + tolerance * scale
    found: list[np.ndarray] = []
    for index in range(values.shape[0]):
        chosen = np.rint(values[index] - first[index])
        rows = [first[index]]
        for column in np.flatnonzero(shortest[index]):
            if not np.array_equal(shifts[column], chosen):
                rows.append(values[index] - shifts[column])
        found.append(np.asarray(rows, dtype=np.float64))
    return found


def _class_key(fraction) -> tuple[int, int, int]:
    """A key of the class of a point modulo the reciprocal lattice (reduced coordinates, steps of 1e-9)."""

    scaled = np.rint((np.asarray(fraction, dtype=np.float64) % 1.0) * 1.0e9).astype(np.int64) % 1_000_000_000
    return tuple(int(value) for value in scaled)


def negation_closed_shortest(fractions, reciprocal, *, tolerance: float = SHORTEST_FOLD_TOLERANCE) -> np.ndarray:
    """One shortest image of each point (:func:`shortest_images`), the set closed under ``k -> -k``.

    The image of :func:`fold_to_shortest`, except at a point of the zone boundary with several shortest images whose
    class ``-f`` came earlier: it takes the negative of that point's image, which is one of its own.  The
    time-reversal pairs of the set (:func:`time_reversal_reduced`) are then exact, ``-k`` and not ``-k`` plus a
    reciprocal vector, which the finite-difference Bloch operator does not keep.  The same points as
    :func:`fold_to_shortest` wherever no point has two shortest images (on a lattice of orthogonal vectors always).
    """

    images = shortest_images(fractions, reciprocal, tolerance=tolerance)
    if not images:
        return np.zeros((0, 3))
    chosen = np.asarray([group[0] for group in images], dtype=np.float64)
    seen: dict = {}
    for index, group in enumerate(images):
        if group.shape[0] > 1:
            partner = seen.get(_class_key(-group[0]))
            if partner is not None:
                chosen[index] = -chosen[partner]
        seen.setdefault(_class_key(group[0]), index)
    return chosen


#: A written k-point ties with its image ``k - G`` where ``|k - G|^2 - |k|^2 <= WRITTEN_TIE_TOLERANCE |G|^2``, twice its
#: distance from the bisector of ``G`` in units of ``|G|``: a zone-boundary point written with four or more decimals.
WRITTEN_TIE_TOLERANCE = 1.0e-3


def written_tie_images(fractions, reciprocal, *, tolerance: float = WRITTEN_TIE_TOLERANCE) -> list[np.ndarray]:
    """The images of each written point that are as short as its shortest image up to the precision of the input.

    ``fractions`` are rows ``f`` of ``k = f B`` (``reciprocal``: the rows of ``B``).  Entry ``i`` is an ``(m, 3)`` array:
    the shortest image ``s`` of the point (:func:`fold_to_shortest`) first, then every ``s - G`` with
    ``|s - G|^2 - |s|^2 <= tolerance |G|^2``.  The test measures the distance from the bisector of each ``G`` in units
    of ``|G|``, so a long cell, whose short reciprocal vectors have close mesh planes, does not join points that are
    not ties, as a test against the largest ``|b_i|^2`` would.  A point written with few decimals is not moved onto
    the boundary: each of its images keeps the rounding of the written point.
    """

    values = np.atleast_2d(fold_to_shortest(fractions, reciprocal))
    if values.shape[0] == 0:
        return []
    basis = np.asarray(reciprocal, dtype=np.float64).reshape(3, 3)
    metric = basis @ basis.T
    radius = math.sqrt(max(float(np.max(np.einsum("...i,ij,...j->...", values, metric, values))), 0.0))
    directions = np.linalg.norm(np.linalg.inv(basis), axis=0)
    # (1 - tolerance) |G|^2 <= 2 s.G bounds |G| by 2 |s| / (1 - tolerance), and each component of G = n B with it
    longest = 2.0 * radius / (1.0 - tolerance)
    reach = [int(math.ceil(longest * float(value) + 1.0e-9)) for value in directions]
    import itertools

    shifts = np.asarray(
        sorted(
            (n for n in itertools.product(*(range(-r, r + 1) for r in reach)) if any(n)),
            key=lambda n: (sum(abs(v) for v in n), n),
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    if shifts.shape[0] == 0:
        return [row[None, :].copy() for row in values]
    lengths = np.einsum("ci,ij,cj->c", shifts, metric, shifts)
    within = lengths <= longest * longest * (1.0 + 1.0e-9)
    shifts, lengths = shifts[within], lengths[within]
    if shifts.shape[0] == 0:
        return [row[None, :].copy() for row in values]
    # |s - G|^2 - |s|^2 = |G|^2 - 2 s.G
    excess = lengths[None, :] - 2.0 * (values @ metric @ shifts.T)
    tied = excess <= tolerance * lengths[None, :]
    found: list[np.ndarray] = []
    for index in range(values.shape[0]):
        rows = shifts[tied[index]]
        found.append(np.vstack([values[index][None, :], values[index][None, :] - rows]))
    return found


def negation_closed_written(k_points, lattice_vectors) -> np.ndarray:
    """The written points of a hybrid's whole mesh (Cartesian) at the images of that mesh, in the written order.

    The image of a point with several shortest images depends on the points before it
    (:func:`negation_closed_shortest`).  Here a component within ``1e-9`` of a half-integer is that half-integer (the
    round trip through Cartesian coordinates leaves ``1/2`` on either side), the points are visited in the order of the
    Monkhorst-Pack mesh (ascending in ``[0, 1)``, the last index fastest), and the images go back to the written rows,
    so neither the order of the lines nor the round-off of the input chooses them.  Of a pair ``k``, ``-k`` the member
    that comes first in the mesh takes the earlier of the two written rows: :func:`time_reversal_reduced` keeps the
    first of a pair, and the exchange of a hybrid is not the same to the order of the discretization at ``k`` and at
    ``-k`` (its pair densities are periodic with other reciprocal vectors).
    """

    points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    reciprocal = reciprocal_lattice(lattice_vectors)
    fractions = points @ np.linalg.inv(reciprocal)
    doubled = 2.0 * fractions
    nearest = np.rint(doubled)
    half = np.abs(doubled - nearest) < 2.0e-9
    fractions = np.where(half, 0.5 * nearest, fractions)
    wrapped = fractions - np.floor(fractions)
    wrapped[np.abs(wrapped - 1.0) < 1.0e-9] = 0.0
    keys = np.rint(wrapped * 1.0e9).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    images = np.zeros_like(wrapped)
    images[order] = negation_closed_shortest(wrapped[order], reciprocal)
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    rows = {tuple(np.rint(row * 1.0e9).astype(np.int64).tolist()): index for index, row in enumerate(images)}
    swapped = np.zeros(order.size, dtype=bool)
    for index in range(order.size):
        if swapped[index]:
            continue
        partner = rows.get(tuple(np.rint(-images[index] * 1.0e9).astype(np.int64).tolist()))
        if partner is None or partner <= index:
            continue
        swapped[partner] = True
        if rank[partner] < rank[index]:
            images[[index, partner]] = images[[partner, index]]
    return images @ reciprocal


def tie_shared_points(
    k_points, weights, lattice_vectors, *, tolerance: float = SHORTEST_FOLD_TOLERANCE, written: bool = False
):
    """``(k_points, weights)`` with every point that has several shortest images (:func:`shortest_images`) replaced
    by all of them, each with an equal share of its weight; the other points keep their bits.

    The whole Monkhorst-Pack mesh of the default routines on an oblique cell, and a ``Kpoint_List`` that
    ``PARSEC_KPOINT_LIST_FOLD`` moves to its shortest images (``SinglePointInput.kpoint_tie_images``): its result does
    not depend on which image a tie would keep, and the set of a whole mesh
    is closed under ``k -> -k`` exactly (the images of ``-f`` are the negatives of those of ``f``), so its
    time-reversal pairs are exact.  ``written``: the points of a ``Kpoint_List``, whose ties are found up to the
    precision of the input (:func:`written_tie_images`).
    """

    k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64)
    reciprocal = reciprocal_lattice(lattice_vectors)
    if written:
        images = written_tie_images(k_points @ np.linalg.inv(reciprocal), reciprocal)
    else:
        images = shortest_images(k_points @ np.linalg.inv(reciprocal), reciprocal, tolerance=tolerance)
    if all(group.shape[0] == 1 for group in images):
        return k_points.copy(), weights.copy()
    rows, shares = [], []
    for point, weight, group in zip(k_points, weights, images):
        if group.shape[0] == 1:
            rows.append(point)
            shares.append(float(weight))
            continue
        for image in group:
            rows.append(image @ reciprocal)
            shares.append(float(weight) / float(group.shape[0]))
    return np.asarray(rows, dtype=np.float64), np.asarray(shares, dtype=np.float64)


def _exact_image_stars(fractional, counts, ekin, grid_map, dims, shifts, rotations, time_reversal, b_rows):
    """The stars of a wedge of shortest representatives made of exact images only, or ``None`` where they are.

    ``irrbz`` counts a mesh point to a star where it is an image of the representative modulo the reciprocal lattice.
    Every shortest image (:func:`shortest_images`) of every member is then ``+-G f_rep`` exactly for an operation
    ``G``, except on the zone boundary, where a member may have several shortest images that the group does not
    relate: those are other k-points of the finite-difference Hamiltonian.  Returns ``None`` where every image is
    exact (the wedge is kept as it is, bit for bit); otherwise ``(fractional, counts, kinetic_energies, grid_map)``
    of the orbits of the images under the group (and time reversal), each image with ``1/m`` of its mesh point: the
    first orbit of each star keeps its representative and ``E_kin``, the others follow it, with ``E_kin = |B f|^2``;
    ``counts`` are the shares (floats) and ``grid_map`` gives each mesh point the orbit of its first image.
    """

    n1, n2, n3 = dims
    full = np.asarray(
        [
            [
                (float(ii) + shifts[0]) / float(n1),
                (float(jj) + shifts[1]) / float(n2),
                (float(ll) + shifts[2]) / float(n3),
            ]
            for ii in range(n1)
            for jj in range(n2)
            for ll in range(n3)
        ],
        dtype=np.float64,
    )
    basis = np.asarray(b_rows, dtype=np.float64)
    images = shortest_images(full, basis)
    stars = np.asarray(grid_map, dtype=np.int64).reshape(-1)
    matrices = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    signs = (1.0, -1.0) if time_reversal else (1.0,)

    def orbit(point):
        turned = np.einsum("rij,j->ri", matrices, np.asarray(point, dtype=np.float64))
        return np.concatenate([sign * turned for sign in signs])

    tolerance = 1.0e-8
    orbits = [orbit(point) for point in np.asarray(fractional, dtype=np.float64)]
    exact = all(
        float(np.min(np.max(np.abs(orbits[stars[index]] - image[None, :]), axis=1))) <= tolerance
        for index, group in enumerate(images)
        for image in group
    )
    if exact:
        return None
    members: dict[int, list[int]] = {}
    for index, star in enumerate(stars):
        members.setdefault(int(star), []).append(index)
    new_fractional, new_counts, new_ekin = [], [], []
    new_map = np.zeros(stars.shape[0], dtype=np.int64)
    for star in range(len(fractional)):
        items = [
            (index, number, image, 1.0 / float(images[index].shape[0]))
            for index in members.get(star, [])
            for number, image in enumerate(images[index])
        ]
        taken = [False] * len(items)
        for first, (index, number, image, share) in enumerate(items):
            if taken[first]:
                continue
            label = len(new_fractional)
            representative = np.asarray(fractional[star], dtype=np.float64) if first == 0 else image
            images_of = orbits[star] if first == 0 else orbit(image)
            total = 0.0
            for other in range(first, len(items)):
                if taken[other]:
                    continue
                candidate = items[other][2]
                if float(np.min(np.max(np.abs(images_of - candidate[None, :]), axis=1))) <= tolerance:
                    taken[other] = True
                    total += items[other][3]
                    if items[other][1] == 0:
                        new_map[items[other][0]] = label
            new_fractional.append(representative)
            new_counts.append(total)
            if first == 0:
                new_ekin.append(float(ekin[star]))
            else:
                cartesian = representative @ basis
                new_ekin.append(float(cartesian @ cartesian))
    return (
        np.asarray(new_fractional, dtype=np.float64),
        np.asarray(new_counts, dtype=np.float64),
        np.asarray(new_ekin, dtype=np.float64),
        new_map.reshape(dims),
    )


def _times(matrix, vector) -> list[float]:
    """``matvec3('N', matrix, vector)`` from ``+0.0``, in Fortran's order."""

    return [
        ((0.0 + matrix[i][0] * vector[0]) + matrix[i][1] * vector[1]) + matrix[i][2] * vector[2]
        for i in range(3)
    ]


def _dot(vector) -> float:
    return ((0.0 + vector[0] * vector[0]) + vector[1] * vector[1]) + vector[2] * vector[2]


def _translate01_rows(values: np.ndarray) -> np.ndarray:
    """:func:`translate01` of every row of ``values`` (elementwise the same)."""

    shifted = np.fmod(values + 0.5, 1.0) - 0.5
    shifted = np.where(np.abs(shifted) < IBZ_TOLERANCE, 0.0, shifted)
    return np.where(shifted < 0.0, shifted + 1.0, shifted)


def _first_equivalent(krot, representatives: np.ndarray, inversions, bvec) -> int | None:
    """The first representative ``k_IBZ`` with ``|B translate01(krot - inv k_IBZ)|^2 < tol``.

    In Fortran's order: representatives in turn, for each ``inv = +1`` then
    ``-1`` (``irrbz.f90:96-127``).
    """

    if representatives.shape[0] == 0:
        return None
    found = None
    for inv in inversions:
        difference = _translate01_rows(np.asarray(krot)[None, :] - inv * representatives)
        cartesian = [
            ((0.0 + bvec[i][0] * difference[:, 0]) + bvec[i][1] * difference[:, 1])
            + bvec[i][2] * difference[:, 2]
            for i in range(3)
        ]
        norm2 = ((0.0 + cartesian[0] * cartesian[0]) + cartesian[1] * cartesian[1]) + (
            cartesian[2] * cartesian[2]
        )
        hits = np.flatnonzero(norm2 < IBZ_TOLERANCE)
        if hits.size and (found is None or hits[0] < found):
            found = int(hits[0])
    return found


@dataclass(frozen=True)
class MonkhorstPackSampling:
    """Fortran PARSEC's Monkhorst-Pack sampling, reduced to the irreducible wedge.

    ``fractional`` are the representatives in reciprocal-lattice
    coordinates (each is the first grid point met of its star), folded into
    ``(-1/2, 1/2]`` (:func:`fold_to_centered_zone`; ``irrbz`` keeps them in
    ``[0, 1)``); ``k_points = B f`` are Cartesian (1/bohr, rows),
    the k-points of the Bloch Hamiltonians.  ``counts`` are the numbers of
    full-grid points each represents, ``weights`` the normalized weights
    ``counts/N`` (renormalized to sum one as ``init_var.F90:365-367``).
    ``kinetic_energies`` are irrbz's ``E_kin`` column (Ry; see
    :func:`irreducible_monkhorst_pack` for what it measures).

    ``grid_map`` is Fortran's ``kptgrid``: ``grid_map[i, j, l]`` is the
    0-based index into ``k_points`` of the full-grid point
    ``((i + s1)/n1, (j + s2)/n2, (l + s3)/n3)``, ``i = 0 .. n1-1`` (Fortran's
    ``kptgrid(i+1, j+1, l+1) - 1``); :meth:`full_grid_fractional` lists those
    points.  The tetrahedron DOS needs it.  ``time_reversal`` says whether
    ``k`` and ``-k`` were identified; ``operations`` is the order of the group
    used.
    """

    dimensions: tuple[int, int, int]
    shift: tuple[float, float, float]
    fractional: np.ndarray
    k_points: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    kinetic_energies: np.ndarray
    grid_map: np.ndarray
    time_reversal: bool = True
    operations: int = 1

    @property
    def count(self) -> int:
        return int(self.k_points.shape[0])

    @property
    def full_grid_size(self) -> int:
        return int(np.prod(self.dimensions))

    @property
    def printed_weights(self) -> np.ndarray:
        """``count/N`` as ``irrbz`` prints it (before the renormalization)."""

        return np.asarray(self.counts, dtype=np.float64) / float(self.full_grid_size)

    def full_grid_fractional(self) -> np.ndarray:
        """``(n1, n2, n3, 3)``: the fractional coordinates of every full-grid point."""

        n1, n2, n3 = self.dimensions
        s1, s2, s3 = self.shift
        axes = [
            (np.arange(n, dtype=np.float64) + s) / float(n)
            for n, s in ((n1, s1), (n2, s2), (n3, s3))
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack(grids, axis=-1)


def irreducible_monkhorst_pack(
    lattice_vectors,
    dimensions,
    shift,
    reciprocal_rotations=None,
    *,
    time_reversal: bool = True,
    fold: bool = True,
    shortest: bool = False,
    ties: bool = False,
) -> MonkhorstPackSampling:
    """Translation of ``irrbz.f90`` (lines 12-213): the IBZ of a shifted MP grid.

    The full grid is ``k = ((i-1) + s_i)/n_i`` (reciprocal-lattice
    coordinates), visited with the last index fastest.  A point is rotated by
    every ``G`` of ``reciprocal_rotations`` (Fortran ``symm%gmtrx``, the
    identity first; ``None`` is the identity alone), folded by
    :func:`translate01`, and compared with every representative found so far,
    directly and (``time_reversal``) with the sign reversed:
    ``|B translate01(G k - inv k_IBZ)|^2 < 1e-8``.  The first match counts the
    point to that representative; a point without one becomes a new
    representative, stored unfolded.  Fortran's "replace by the shortest
    member" branch is kept as written, which makes it dead (``norm2`` has
    ``+ 8`` added): the representative is the first member met.  With
    ``fold`` (the default, a decision of this port) the representatives are
    then folded into ``(-1/2, 1/2]`` (:func:`fold_to_centered_zone`), which
    PARSEC 1.5 intends and does not do: its k-points stay in ``[0, 1)``, so a
    run without symmetry differs from the symmetric one through the
    finite-difference Bloch operator (Si8, MP 2x2x2 shifted, h 0.64: 15 mRy).
    The counts, weights, ``grid_map`` and ``E_kin`` do not depend on it;
    ``fold=False`` gives PARSEC 1.5's set.  ``shortest`` (with ``fold``; the
    default routines under ``PARSEC_KPOINT_FOLD=shortest`` and a mesh under
    ``PARSEC_KPOINT_MESH_GUARD``,
    ``SinglePointInput.kpoint_shortest_representatives``) moves each folded
    representative on to its shortest image modulo the reciprocal lattice
    (:func:`fold_to_shortest`), which differs from the folded one on an
    oblique cell only.  ``ties`` (with ``shortest``; the default routines'
    wedges of scalar, collinear-spin and spin-orbit orbitals) splits a star
    whose members' shortest images (:func:`shortest_images`, all of them on
    the zone boundary, each with an equal share of its point) are not all
    exact images ``+-G f`` of its representative into the orbits of those
    images (:func:`_exact_image_stars`): the wedge then gives the whole mesh
    of shared ties (:func:`tie_shared_points`) to round-off; ``counts`` are
    floats there and ``grid_map`` gives a mesh point the orbit of its first
    image.  A wedge whose stars are exact is kept bit for bit.  ``E_kin``
    of a new representative is ``|B (k' - anint(k'))|^2`` with ``k'`` the
    point rotated by the *last* operation, as the Fortran computes it.

    ``lattice_vectors`` are rows in bohr; ``B`` is the reciprocal lattice of
    ``inipbc.f90`` (:func:`~parsec_python.Grid.space_group.fortran_reciprocal_lattice`).
    Raises ``ValueError`` where the counts do not add up to the grid (PARSEC
    error 241).
    """

    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    shifts = tuple(float(value) for value in shift)
    if len(shifts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in shifts):
        raise ValueError("the Monkhorst-Pack shift must be three non-negative numbers")
    if reciprocal_rotations is None:
        rotations = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    else:
        rotations = [
            [[float(value) for value in row] for row in matrix]
            for matrix in np.asarray(reciprocal_rotations).reshape(-1, 3, 3)
        ]
    # bvec(i, j): component i of reciprocal vector j; rows of B are the vectors.
    b_rows = fortran_reciprocal_lattice(lattice_vectors)
    bvec = [[float(b_rows[j][i]) for j in range(3)] for i in range(3)]
    n1, n2, n3 = dims
    kibz: list[list[float]] = []
    counts: list[int] = []
    ekin: list[float] = []
    grid_map = np.zeros(dims, dtype=np.int64)
    inversions = (1.0, -1.0) if time_reversal else (1.0,)
    # The representatives so far, one row each, for the comparisons below
    # (elementwise the same arithmetic as the scalar Fortran loop).
    stored = np.zeros((n1 * n2 * n3 + 1, 3))
    for ii in range(n1):
        for jj in range(n2):
            for ll in range(n3):
                kprot = [
                    (float(ii) + shifts[0]) / float(n1),
                    (float(jj) + shifts[1]) / float(n2),
                    (float(ll) + shifts[2]) / float(n3),
                ]
                new_point = True
                krot = kprot
                for gmtrx in rotations:
                    krot = translate01(_times(gmtrx, kprot))
                    if not kibz:
                        kibz.append(list(krot))
                        stored[0] = krot
                        ktest = [value - fortran_anint(value) for value in krot]
                        ekin.append(_dot(_times(bvec, ktest)))
                        counts.append(0)
                        grid_map[ii, jj, ll] = 0
                    itest = _first_equivalent(krot, stored[: len(kibz)], inversions, bvec)
                    if itest is None:
                        continue
                    # The first match ends the search: Fortran goes on through
                    # the remaining operations with new_kpt false, which
                    # changes nothing (the E_kin of the last operation is
                    # used for new points only).
                    new_point = False
                    ktest = [value - fortran_anint(value) for value in kprot]
                    norm2 = _dot(_times(bvec, ktest)) + 8.0
                    grid_map[ii, jj, ll] = itest
                    norm1 = _dot(_times(bvec, kibz[itest]))
                    if norm2 < norm1:
                        kibz[itest] = translate01(kprot)
                        stored[itest] = kibz[itest]
                        ekin[itest] = norm2
                    counts[itest] += 1
                    break
                if new_point:
                    kibz.append(list(kprot))
                    stored[len(kibz) - 1] = kprot
                    ktest = [value - fortran_anint(value) for value in krot]
                    ekin.append(_dot(_times(bvec, ktest)))
                    counts.append(1)
                    grid_map[ii, jj, ll] = len(kibz) - 1
    total = n1 * n2 * n3
    if sum(counts) != total:
        raise ValueError(
            f"incorrect Brillouin zone: {sum(counts)} of {total} grid points "
            "counted (PARSEC error 241)"
        )
    fractional = np.array(kibz, dtype=np.float64)
    count_values = np.asarray(counts, dtype=np.int64)
    if fold and shortest:
        # the shortest image of each representative (the Wigner-Seitz cell; the folded point where it is one)
        fractional = fold_to_shortest(fractional, np.asarray(b_rows, dtype=np.float64))
        if ties and oblique_reciprocal_lattice(np.asarray(b_rows, dtype=np.float64)):
            # the stars of exact images (the ties on the zone boundary shared among their images)
            split = _exact_image_stars(
                fractional, counts, ekin, grid_map, dims, shifts, rotations, time_reversal, b_rows
            )
            if split is not None:
                fractional, count_values, split_ekin, grid_map = split
                ekin = list(split_ekin)
    elif fold:
        # the representatives themselves (the search above compares them modulo G, as irrbz does)
        fractional = fold_to_centered_zone(fractional)
    k_points = np.array([_times(bvec, [float(v) for v in point]) for point in fractional], dtype=np.float64)
    weights = np.array([float(count) / float(total) for count in count_values], dtype=np.float64)
    norm = 0.0
    for value in weights:
        norm = norm + float(value)
    weights = weights / norm
    return MonkhorstPackSampling(
        dimensions=dims,
        shift=shifts,
        fractional=fractional,
        k_points=k_points,
        weights=weights,
        counts=count_values,
        kinetic_energies=np.asarray(ekin, dtype=np.float64),
        grid_map=grid_map,
        time_reversal=bool(time_reversal),
        operations=len(rotations),
    )


def mesh_preserving_operations(group, dimensions, shift, *, tolerance: float = 1.0e-8):
    """The operations of ``group`` whose rotation maps a Monkhorst-Pack mesh onto itself, as a ``SpaceGroup``.

    The mesh point ``f = (i + s)/n`` (reciprocal-lattice coordinates) goes to ``G f`` with ``G = transpose(M)``
    (:attr:`~parsec_python.Grid.space_group.SpaceGroup.reciprocal_rotations`), a mesh point again (modulo the
    reciprocal lattice) when ``n (G f) - s`` is a whole vector for every ``i``: an affine function of ``i``, tested
    on the corners of one mesh cell.  ``irrbz`` reduces any mesh by the whole group and the density average then
    restores the star of each representative, also its members outside the mesh, the sampling of the symmetrized
    mesh.  With the operations kept here every star lies in the mesh, so the wedge with the averaged density gives
    the density of the full mesh (time reversal needs no such test: ``k`` and ``-k`` have the same density, and a
    star ``-G k`` with one member in a mesh that ``G`` preserves lies in it entirely).  Returns ``group`` itself when
    every operation is kept.
    """

    from .space_group import SYMMETRY_TOLERANCE, SpaceGroup

    counts = np.asarray([int(value) for value in dimensions], dtype=np.float64)
    offsets = np.asarray([float(value) for value in shift], dtype=np.float64)
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64)
    points = (corners + offsets) / counts
    kept = []
    for index, rotation in enumerate(group.reciprocal_rotations):
        target = (points @ rotation.T.astype(np.float64)) * counts - offsets
        if np.max(np.abs(target - np.rint(target))) <= tolerance:
            kept.append(index)
    if len(kept) == group.order:
        return group
    last = group.translations[kept[-1]]
    return SpaceGroup(
        rotations=tuple(group.rotations[index] for index in kept),
        translations=tuple(group.translations[index] for index in kept),
        cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
        labels=tuple(group.labels[index] for index in kept),
        crystal_system=group.crystal_system,
        full_holohedry=False,
        lattice_operations=group.lattice_operations,
        candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
        symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
        ignored=group.ignored,
    )


def list_preserving_operations(group, k_points, weights, lattice_vectors, *, tolerance: float = 1.0e-6):
    """The operations of ``group`` whose rotation maps a weighted k-point list onto itself, as a ``SpaceGroup``.

    ``k_points`` are Cartesian (1/bohr), ``weights`` their weights.  The list is taken as the weights of its classes
    ``{k, -k}`` modulo the reciprocal lattice (time reversal: ``k`` and ``-k`` give the same density and forces for
    scalar and collinear-spin orbitals); an operation is kept where ``G f`` (``G`` its
    :attr:`~parsec_python.Grid.space_group.SpaceGroup.reciprocal_rotations`, ``f`` the reciprocal-lattice coordinates)
    of every class is a class of the same total weight (relative ``1e-8``).  Coordinates compare to ``tolerance``, the
    precision of a written list.  Returns ``group`` itself when every operation is kept.
    """

    from .space_group import SYMMETRY_TOLERANCE, SpaceGroup

    vectors = reciprocal_lattice(np.asarray(lattice_vectors, dtype=float))
    fractions = np.atleast_2d(np.asarray(k_points, dtype=float)) @ np.linalg.inv(vectors)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    scale = 1.0 / float(tolerance)

    def key(fraction) -> tuple:
        # the class of f: the smaller (lexicographically) of f and -f, both reduced to [0, 1) on the tolerance's grid
        candidates = []
        for value in (fraction, -fraction):
            whole = np.rint(np.asarray(value) * scale).astype(np.int64)
            candidates.append(tuple(int(item) for item in np.mod(whole, int(round(scale)))))
        return min(candidates)

    classes: dict[tuple, float] = {}
    for fraction, weight in zip(fractions, weights):
        classes[key(fraction)] = classes.get(key(fraction), 0.0) + float(weight)
    representatives = {}
    for fraction in fractions:
        representatives.setdefault(key(fraction), fraction)
    total = float(np.sum(weights))
    kept = []
    for index, rotation in enumerate(group.reciprocal_rotations):
        matrix = np.asarray(rotation, dtype=np.float64)
        keeps = True
        for label, fraction in representatives.items():
            image = classes.get(key(fraction @ matrix.T))
            if image is None or abs(image - classes[label]) > 1.0e-8 * max(total, 1.0e-300):
                keeps = False
                break
        if keeps:
            kept.append(index)
    if len(kept) == group.order:
        return group
    last = group.translations[kept[-1]]
    return SpaceGroup(
        rotations=tuple(group.rotations[index] for index in kept),
        translations=tuple(group.translations[index] for index in kept),
        cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
        labels=tuple(group.labels[index] for index in kept),
        crystal_system=group.crystal_system,
        full_holohedry=False,
        lattice_operations=group.lattice_operations,
        candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
        symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
        ignored=group.ignored,
    )


def default_stencil_lines(lattice_vectors, counts) -> list[tuple[tuple[int, int, int], float]]:
    """The second differences ``((offset, weight), ...)`` of the Laplacian of the default routines' grid of
    ``counts`` points along ``lattice_vectors`` (offsets in grid steps): the three axes, then the extra directions of
    an oblique grid (:func:`parsec_python.Laplacian.lattice_metric.second_derivative_plan`)."""

    from ..Laplacian.lattice_metric import second_derivative_plan

    counts = np.asarray([float(value) for value in counts], dtype=np.float64)
    steps = np.asarray(lattice_vectors, dtype=np.float64) / counts[:, None]
    plan = second_derivative_plan(steps)
    lines = [(tuple(int(v) for v in np.eye(3, dtype=int)[axis]), float(plan.axial[axis])) for axis in range(3)]
    lines += [
        (tuple(int(v) for v in direction), float(weight)) for direction, weight in zip(plan.directions, plan.weights)
    ]
    return lines


def grid_stencil_lines(cell, settings):
    """``(counts, lines)`` of the grid that the routines of ``settings`` build for the bulk ``cell``: its points along
    the lattice vectors and the second differences ``((offset, weight), ...)`` of its Laplacian (offsets in grid
    steps, weights in 1/bohr^2), for :func:`stencil_preserving_matrices`.

    The default routines: :func:`default_stencil_lines` of
    :func:`parsec_python.Grid.pbc.default_periodic_grid_counts`.  The Fortran-derived routines
    (``settings.fortran_routines``): Fortran's counts (:func:`parsec_python.Grid.pbc.fortran_grid_counts`), on the
    orthorhombic grid the three axes of one step, on Fortran's general grid of a non-orthogonal cell the axes and the
    auxiliary directions of ``lapmvs`` with their ``b / step^2``
    (:func:`parsec_python.Laplacian.nonorthogonal.pbc_grid_coefficients`).  Both gradients are the centred
    differences along the three axes.
    """

    from .pbc import default_periodic_grid_counts, fortran_grid_counts, fortran_orthorhombic_counts

    lattice_vectors = np.asarray(cell.lattice_vectors, dtype=np.float64)
    if not getattr(settings, "fortran_routines", False):
        counts = np.asarray(default_periodic_grid_counts(cell, settings), dtype=np.int64)
        return counts, default_stencil_lines(lattice_vectors, counts)
    counts, steps = fortran_grid_counts(cell, settings)
    counts = np.asarray(counts, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.float64)
    if fortran_orthorhombic_counts(cell, settings) is not None:
        spacing = float(np.diag(lattice_vectors)[0]) / float(counts[0])
        return counts, [(tuple(int(v) for v in np.eye(3, dtype=int)[axis]), 1.0 / spacing**2) for axis in range(3)]
    from ..Laplacian.nonorthogonal import pbc_grid_coefficients

    coefficients = pbc_grid_coefficients(lattice_vectors, steps)
    return counts, [
        (tuple(int(v) for v in offset), float(coefficient) / float(step) ** 2)
        for offset, coefficient, step in coefficients.stencil_terms(steps)
    ]


def stencil_preserving_matrices(matrices, counts, lines, *, tolerance: float = 1.0e-9) -> list[bool]:
    """For each integer matrix ``M`` (lattice coordinates) of ``matrices``: whether the operation maps the
    finite-difference operators of the grid of ``counts`` points (second differences ``lines``, as
    :func:`grid_stencil_lines` gives them; centred first differences along the axes) onto themselves.

    ``W = N M N^-1`` (``N = diag(counts)``) moves a stencil offset ``o`` to ``W o``: kept where ``W`` is a signed
    permutation of the axes (the gradient) that maps every second difference onto one of the same weight (see
    :func:`stencil_preserving_operations`).
    """

    counts = np.asarray([float(value) for value in counts], dtype=np.float64)
    scale = max(abs(weight) for _, weight in lines)
    weights = {}
    for offset, weight in lines:
        weights[tuple(int(v) for v in offset)] = weight
        weights[tuple(-int(v) for v in offset)] = weight
    flags = []
    for matrix in np.asarray(matrices).reshape(-1, 3, 3):
        index_map = counts[:, None] * np.asarray(matrix, dtype=np.float64) / counts[None, :]
        rounded = np.rint(index_map)
        if np.max(np.abs(index_map - rounded)) > 1.0e-9:
            flags.append(False)
            continue
        rounded = rounded.astype(np.int64)
        if not (np.all(np.abs(rounded).sum(axis=0) == 1) and np.all(np.abs(rounded).sum(axis=1) == 1)):
            flags.append(False)
            continue
        preserved = True
        for offset, weight in lines:
            image = tuple(int(v) for v in rounded @ np.asarray(offset, dtype=np.int64))
            other = weights.get(image)
            if other is None or abs(other - weight) > tolerance * scale:
                preserved = False
                break
        flags.append(preserved)
    return flags


def stencil_preserving_operations(group, lattice_vectors, counts, *, tolerance: float = 1.0e-9, lines=None):
    """The operations of ``group`` that map the finite-difference operators of the default routines' grid onto
    themselves, as a ``SpaceGroup`` (``group`` itself when every operation is kept).

    The grid of the default routines has ``counts`` points along the lattice vectors (steps ``s_i = a_i/n_i``).  Its
    gradient is the centred first difference along the three grid axes, turned Cartesian by ``S^-1``
    (:func:`parsec_python.Laplacian.finite_difference.build_gradient`), and its Laplacian the second differences
    along the axes and, on an oblique grid, along the extra directions of
    :func:`parsec_python.Laplacian.lattice_metric.second_derivative_plan`, each with its weight.  An operation with
    the integer matrix ``M`` (lattice coordinates) moves a stencil offset ``o`` (grid steps) to ``W o`` with
    ``W = N M N^-1``, ``N = diag(counts)``.  The discrete Hamiltonian ``-nabla^2 - 2ik.nabla + k^2`` of a k-point
    commutes with the operation (``k`` turned with it) when ``W`` is a signed permutation of the axes (the gradient's
    offsets) that maps every second difference onto one of the same weight; these are the operations kept.  On an
    orthogonal grid every operation that maps the grid onto itself is kept; on an oblique one an operation that turns
    an axis onto an extra direction (the three- and six-fold axes of a hexagonal cell, the three-fold axes of the
    primitive fcc cell off its body diagonal) changes the operators at the order of the discretization, and the
    wedge of such operations is not the energy of the same points without them (hcp Au2, 0.3 bohr: 7.7e-3 Ry).  hcp
    keeps 8 of its 24 operations, the primitive fcc cell 12 of 48; their wedges gave the energy of time reversal
    alone on the same points within 1e-10 and 2e-9 Ry (A100, 2026-11-19).  The kept operations form a group (the
    operations that preserve a set of offsets and weights do).  ``lines`` (:func:`grid_stencil_lines`) gives the
    second differences of another grid (Fortran's general grid); ``None`` those of the default routines.
    """

    from .space_group import SYMMETRY_TOLERANCE, SpaceGroup

    if lines is None:
        lines = default_stencil_lines(lattice_vectors, counts)
    preserved = stencil_preserving_matrices(group.rotation_matrices, counts, lines, tolerance=tolerance)
    kept = [index for index, flag in enumerate(preserved) if flag]
    if len(kept) == group.order:
        return group
    last = group.translations[kept[-1]]
    return SpaceGroup(
        rotations=tuple(group.rotations[index] for index in kept),
        translations=tuple(group.translations[index] for index in kept),
        cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
        labels=tuple(group.labels[index] for index in kept),
        crystal_system=group.crystal_system,
        full_holohedry=False,
        lattice_operations=group.lattice_operations,
        candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
        symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
        ignored=group.ignored,
    )


def kpoint_sampling_details(problem) -> MonkhorstPackSampling | None:
    """The Fortran-path sampling of a ``SinglePointInput``, or ``None``.

    Only an input with ``monkhorst_pack_shift`` (PARSEC's
    ``Monkhorst_Pack_Shift`` block, or zero for a ``Monkhorst_Pack_Grid``
    without it, :func:`unshifted_mesh_setting`) takes it: the grid of
    ``monkhorst_pack_dimensions`` with that shift, reduced by the input's
    ``space_group`` (the identity where it is ``None``, e.g. with
    ``Ignore_Symmetry``) and by time reversal unless the calculation is
    non-collinear (``irrbz.f90:96-99``).  A hybrid functional takes time
    reversal only (its exchange operator sums over the full mesh,
    :mod:`parsec_python.SCF.hybrid_periodic`), or with
    ``problem.hybrid_kpoint_wedge`` (``PARSEC_HYBRID_KPOINT_WEDGE``) the
    operations of the space group that map the mesh onto itself
    (:func:`hybrid_wedge_group`): the exchange then reaches the other points
    of the mesh through the images of the wedge.  The representatives are
    folded into ``(-1/2, 1/2]``, or moved on to their shortest images
    (:func:`fold_to_shortest`) where
    ``problem.kpoint_shortest_representatives`` (the default routines under
    ``PARSEC_KPOINT_FOLD=shortest``); with ``problem.kpoint_tie_images``
    (not for a hybrid, whose exchange reaches the mesh through the stars of
    its wedge, nor for the DOS tetrahedra and Output_GW) a star whose members'
    shortest images are not all exact images of its representative is split
    into their orbits (``ties`` of :func:`irreducible_monkhorst_pack`).
    """

    shift = getattr(problem, "monkhorst_pack_shift", None)
    if shift is None or problem.monkhorst_pack_dimensions is None:
        return None
    if _hybrid(problem):
        group = hybrid_wedge_group(problem) if getattr(problem, "hybrid_kpoint_wedge", False) else None
    else:
        group = getattr(problem, "space_group", None)
    lattice = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
    )
    rotations = (
        None
        if group is None
        else tuple(tuple(tuple(int(v) for v in row) for row in matrix) for matrix in group.reciprocal_rotations)
    )
    return _cached_irreducible_monkhorst_pack(
        lattice,
        tuple(problem.monkhorst_pack_dimensions),
        tuple(float(value) for value in shift),
        rotations,
        not problem.scf.noncollinear,
        bool(getattr(problem, "kpoint_shortest_representatives", False)),
        bool(getattr(problem, "kpoint_tie_images", False)) and not _hybrid(problem),
    )


@lru_cache(maxsize=8)
def _cached_irreducible_monkhorst_pack(
    lattice, dimensions, shift, rotations, time_reversal, shortest=False, ties=False
):
    """One computation per input for the CLI, the report and ``qpoints.dat``.

    Callers must not modify the arrays of the returned (shared) object.
    """

    sampling = irreducible_monkhorst_pack(
        np.asarray(lattice, dtype=float),
        dimensions,
        shift,
        None if rotations is None else np.asarray(rotations, dtype=np.int64),
        time_reversal=time_reversal,
        shortest=bool(shortest),
        ties=bool(ties),
    )
    for array in (
        sampling.fractional,
        sampling.k_points,
        sampling.weights,
        sampling.counts,
        sampling.kinetic_energies,
        sampling.grid_map,
    ):
        array.flags.writeable = False
    return sampling


def _hybrid(problem) -> bool:
    return bool(getattr(getattr(problem, "scf", None), "is_hybrid", False))


HYBRID_KPOINT_WEDGE_ENV = "PARSEC_HYBRID_KPOINT_WEDGE"
#: The value of PARSEC_HYBRID_KPOINT_WEDGE without the variable (since 2026-11-23; ``auto`` before).
HYBRID_KPOINT_WEDGE_DEFAULT = "on"


def hybrid_kpoint_wedge_setting() -> str:
    """``PARSEC_HYBRID_KPOINT_WEDGE`` (``auto``, ``on``, ``off``): the irreducible wedge of the Monkhorst-Pack mesh of a
    hybrid single point of a bulk cell, under the operations of the space group that map the grid, the mesh and the
    grid's finite-difference operators onto themselves (:func:`hybrid_wedge_operations`, on any grid) and time
    reversal, with the density symmetrized over them: the exchange sums over the whole mesh through the images of the
    k-points solved, their Bloch factors with the grid points permuted (Quantum ESPRESSO's ``exxinit`` rotates its
    wavefunctions the same way; :class:`parsec_python.V_xc.bloch_exchange.MeshStars`).

    * ``on`` (default, since 2026-11-23): the wedge always, its semilocal start too, metals included: the members of
      every star of the mesh are filled alike (within the little group of a k-point the wedge's own Fock loop can
      still break: fcc Au 3x3x3 and 6x6x6 at 1000 K became unstable at outer iterations 13 and 8).  The lower state
      that the time-reversal pairs of a metal reach at a small smearing is an artefact of the finite mesh: exact
      exchange favours integer occupations of a level that is partly filled at the Fermi level at a set of equivalent
      k-points (the exchange of a state at one k-point with itself, ``f^2 w_k^2`` times the kernel's element at
      q + G = 0, which grows only as ``N_k^(2/3)``: a gain of order ``N_k^(-4/3)`` per level), so a Fock loop of the
      pairs that runs long enough moves the electron of such a level onto one of the points and breaks the crystal
      symmetry.  The split of a filled and an emptied member per unit of occupation is that self-exchange,
      ``a_x K(q+G=0) / (N_k Omega)`` (fcc Au: 0.0638, 0.0425, 0.0319 Ry on 4x4x4, 6x6x6, 8x8x8; measured 0.0637,
      0.0415, 0.0326 Ry), which falls as ``N_k^(-1/3)``; a level breaks only where it times ``f (1 - f)`` outweighs
      about kT, so the instability leaves any finite temperature as the mesh grows, slowly at the default 80 K (about
      100 points per direction by that estimate).  Quantum ESPRESSO's exchange through the rotated orbitals of its
      irreducible points gives the symmetric state.  A100, PBE0, primitive fcc Au (the PBE table of the repository,
      0.25 bohr, order 12), ``F = E - TS`` of the pairs minus that of the wedge: at 80 K and ``Exx_Tolerance`` 1e-10
      -2.28e-3 Ry on 3x3x3, -6.33e-4 Ry on 4x4x4 (the pairs unstable at outer iteration 4, integer occupations),
      -3.06e-4 Ry on 6x6x6; at 1000 K -1.21e-3 Ry on 3x3x3, -2.89e-4 Ry on 4x4x4 (one of the three X points takes the
      level, f 0.99 against 0.005), -8.3e-5 Ry on 6x6x6, -5.7e-7 Ry on 8x8x8 (the pairs stay symmetric down to dexx
      8e-14); F of the wedge moves by 1.06e-2 Ry from 4x4x4 to 6x6x6 at 1000 K.  On 4x4x4 at 1500 K -1.22e-4 Ry, at
      2000 K none (the pairs converge to dexx 1e-14 on the symmetric state).  The symmetric state of the same PBE
      table has its X level at E_F + 0.0044 Ry, f 0.332, in this code and in Quantum ESPRESSO 7.5 alike, and Quantum
      ESPRESSO's time-reversal pairs (``nosym``) break to the same levels and F (2.886e-4 Ry lower).  Insulators take
      the same state as under ``auto``, 12 to 31 % faster (the start on the wedge too).

      The limit: the forces of the wedge are the slope of F for displacements that keep a subgroup of the wedge's
      operations.  A structure of lower symmetry runs on the time-reversal pairs (and so do molecular dynamics,
      ``manual`` movements, spin with forces, a ``Kpoint_List``, slabs and spinors), whose Fock loop can end in the
      broken state, so at a tight ``Exx_Tolerance``, a coarse mesh and a low smearing its energy is not comparable with
      that of a symmetric structure: fcc Au at 80 K on 4x4x4, the crystal translated by (1e-4, 2e-4, 0) bohr lies
      6.33e-4 Ry below the symmetric one at ``Exx_Tolerance`` 1e-10 (7.6e-7 Ry at the default 1e-6, where the pairs
      stop before the instability; with the CA table at 0.35 bohr and order 6 they break at 1e-6 too, E 6.6e-4 Ry
      below the wedge).  ``parsec.out`` names the state a metal's Fock loop ends in
      (:meth:`parsec_python.SCF.hybrid_periodic.BlochExchange.after_loop`).
    * ``auto`` (the default from 2026-11-20 to 2026-11-22): the semilocal start runs on the time-reversal pairs, bit
      for bit as ``off``; where it fills every state to 0 or 1 (no occupation within
      :data:`HYBRID_WEDGE_FRACTIONAL_TOLERANCE` of the Fermi-Dirac interior, :func:`partly_filled_states`: an
      insulator) the Fock loop moves to the wedge, from the representatives' orbitals of that start
      (:meth:`parsec_python.SCF.hybrid_periodic.BlochExchange.after_start`); a start with a partly filled state keeps
      the pairs for the whole run, bit for bit as ``off``.
    * ``off``: the time-reversal pairs of the mesh, as before.

    Forces and moving atoms take the same route unless ``PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off``
    (:func:`hybrid_kpoint_force_wedge_setting`); spinors keep the pairs either way.  Any other value is refused (``grid``, the operations
    of scalar orbitals on any grid of 2026-11-19, among them those that turn an oblique grid's stencil, is gone)."""

    import os

    value = os.environ.get(HYBRID_KPOINT_WEDGE_ENV, HYBRID_KPOINT_WEDGE_DEFAULT).strip().lower()
    value = value or HYBRID_KPOINT_WEDGE_DEFAULT
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{HYBRID_KPOINT_WEDGE_ENV} must be auto, on or off, got {value!r}")
    return value


HYBRID_KPOINT_FORCE_WEDGE_ENV = "PARSEC_HYBRID_KPOINT_FORCE_WEDGE"


def hybrid_kpoint_force_wedge_setting() -> str:
    """``PARSEC_HYBRID_KPOINT_FORCE_WEDGE`` (``auto``, ``off``): the forces and the relaxation of a hybrid on the
    irreducible wedge of its Monkhorst-Pack mesh.

    * ``auto`` (default, since 2026-11-21): the single-point forces and the ``bfgs``/``simple`` relaxations of a bulk
      hybrid take the wedge that :func:`hybrid_kpoint_wedge_setting` gives a single point (``on``, its default since
      2026-11-23: the whole run on it, metals too; ``auto``: after a semilocal start of the first structure without
      partly filled states, which the later structures of a relaxation keep, so a metal keeps the pairs), under the
      operations of the space group that map the grid, its finite-difference operators and the mesh onto themselves
      and keep the movable atoms apart from the fixed ones; the nonlocal forces of the wedge are completed over the
      group and the forces averaged over it (``symm_forces.f90``), so the structure keeps the group, as for semilocal
      k-point forces.  On the default
      routines ``PARSEC_KPOINT_FORCE_WEDGE=off`` (:func:`kpoint_force_wedge_setting`) turns this off too; the
      Fortran-derived routines do not read that switch (they reduce the mesh of every k-point force, as PARSEC 1.5
      does), so there ``off`` of this switch is the one that keeps a hybrid's forces on the pairs.  Molecular
      dynamics and ``manual`` movements keep the pairs (PARSEC uses no symmetry for them).
    * ``off``: forces, moving atoms and relaxations of a hybrid keep the time-reversal pairs of the whole mesh, as
      before, bit for bit.

    Any other value is refused."""

    import os

    value = os.environ.get(HYBRID_KPOINT_FORCE_WEDGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{HYBRID_KPOINT_FORCE_WEDGE_ENV} must be auto or off, got {value!r}")
    return value


#: A state of the semilocal start of a hybrid is partly filled where ``tol <= f <= 1 - tol`` (``f`` per spin orbital,
#: 0 to 1), ``tol`` this value (:func:`partly_filled_states`, ``PARSEC_HYBRID_KPOINT_WEDGE=auto``).  Far above the
#: round-off of the occupations (they are exactly 0 or 1 beyond 36.7 kT from the Fermi level, the cut-off of
#: ``flevel.f90``, :func:`parsec_python.Occupations.fermi_dirac.fermi_occupations`), and small enough that any level
#: within 13.8 kT of the Fermi level counts (f >= 1e-6), also the tail of a level the Fock loop may pull across it.
#: An insulator at Fermi_Temp 1 K (kT = 6.3e-6 Ry) is exactly 0/1, and so is one at the default 80 K whose gap
#: exceeds about 0.014 Ry (27.6 kT, the Fermi level mid-gap); a metal's
#: levels at the Fermi level are partly filled at any smearing (unless its mesh puts no level within 13.8 kT of
#: it: the test sees the semilocal start only).  Only speed is lost where a smeared insulator counts (the pairs are
#: the exact route): at 1000 K a gap below about 0.17 Ry keeps the pairs.
HYBRID_WEDGE_FRACTIONAL_TOLERANCE = 1.0e-6


def partly_filled_states(occupations, tolerance: float = HYBRID_WEDGE_FRACTIONAL_TOLERANCE) -> tuple[int, float]:
    """``(count, largest)`` of the partly filled states among ``occupations`` (a sequence of arrays, one per spin
    channel or k-point, or one array; values per spin orbital in ``[0, 1]``): ``count`` those with
    ``tolerance <= f <= 1 - tolerance``, ``largest`` the largest ``min(f, 1 - f)`` of all (0 where every state is
    exactly 0 or 1)."""

    tolerance = float(tolerance)
    if not 0.0 < tolerance < 0.5:
        raise ValueError("the tolerance of a partly filled state must lie between 0 and 1/2")
    if isinstance(occupations, np.ndarray):
        occupations = [occupations]
    count, largest = 0, 0.0
    for values in occupations:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            continue
        count += int(np.count_nonzero((values >= tolerance) & (values <= 1.0 - tolerance)))
        largest = max(largest, float(np.max(np.minimum(values, 1.0 - values))))
    return count, max(largest, 0.0)


@dataclass(frozen=True)
class DeferredHybridWedge:
    """The wedge a hybrid single point takes under ``PARSEC_HYBRID_KPOINT_WEDGE=auto`` if its semilocal start fills
    every state to 0 or 1 (``SinglePointInput.hybrid_kpoint_wedge_deferred``; the input is otherwise that of ``off``).

    ``group`` the space group of the wedge (:func:`hybrid_wedge_operations`: the stencil-exact operations that keep
    the grid and the mesh ``(dimensions, shift)``); ``density_group`` the group the wedge's density is averaged over
    where the input of the pairs averages none (the default routines: ``group``), ``None`` where the wedge keeps the
    input's average (the Fortran-derived routines average over the whole space group; the images take the
    stencil-exact operations of it)."""

    group: object
    dimensions: tuple
    shift: tuple
    density_group: object = None


def mesh_keeps_time_reversal(shift, *, tolerance: float = 1.0e-8) -> bool:
    """Whether ``k -> -k`` maps the Monkhorst-Pack mesh ``(i + s)/n`` onto itself: ``-(i + s) - s`` whole for every
    ``i``, so ``2 s`` whole in every direction (a shift of 0 or 1/2).  The exchange of a hybrid's time-reversal pairs
    sums over the mesh and its mirror image; its irreducible wedge (``PARSEC_HYBRID_KPOINT_WEDGE``) is taken only for
    a mesh that is its own mirror image, where both are the mesh."""

    doubled = 2.0 * np.asarray([float(value) for value in shift], dtype=np.float64)
    return bool(np.max(np.abs(doubled - np.rint(doubled)), initial=0.0) <= tolerance)


def grid_is_oblique(cell, settings) -> bool:
    """Whether the grid the routines of ``settings`` build for the bulk ``cell`` has second differences beyond its
    three axes (:func:`grid_stencil_lines`: the extra directions of an oblique default-routines grid, the auxiliary
    directions of Fortran's general grid of a non-orthogonal cell)."""

    return len(grid_stencil_lines(cell, settings)[1]) > 3


def hybrid_wedge_operations(group, cell, settings, dimensions, shift):
    """The operations of ``group`` for the irreducible wedge of a hybrid's Monkhorst-Pack mesh ``(dimensions,
    shift)`` of the bulk ``cell`` on the grid of ``settings``, or ``None`` (none but the identity, or the pairs).

    Those that map the mesh onto itself (:func:`mesh_preserving_operations`; ``None`` for a mesh that time reversal
    does not keep, :func:`mesh_keeps_time_reversal`) and the grid's finite-difference operators onto themselves
    (:func:`stencil_preserving_operations` with the stencil of :func:`grid_stencil_lines`, on any grid: on an oblique
    grid the others, the three- and six-fold axes of a hexagonal cell, change the discrete Hamiltonian at the order
    of the discretization; on an orthogonal grid every mesh-keeping operation is kept).  The exchange images take the
    same operations (:func:`parsec_python.SCF.hybrid_periodic.hybrid_mesh_stars`).  Whether a run takes the wedge is
    :func:`hybrid_kpoint_wedge_setting`'s (``on``; ``auto`` after a semilocal start without partly filled states).
    """

    if group is None or shift is None or dimensions is None or not mesh_keeps_time_reversal(shift):
        return None
    kept = mesh_preserving_operations(group, dimensions, shift)
    if kept.order > 1 and cell is not None and settings is not None:
        counts, lines = grid_stencil_lines(cell, settings)
        kept = stencil_preserving_operations(
            kept, np.asarray(cell.lattice_vectors, dtype=np.float64), counts, lines=lines
        )
    return kept if kept.order > 1 else None


def hybrid_wedge_group(problem):
    """The operations of ``problem.space_group`` for the irreducible wedge of a hybrid (``SinglePointInput
    .hybrid_kpoint_wedge``): :func:`hybrid_wedge_operations` of its mesh and grid, or ``None``."""

    return hybrid_wedge_operations(
        getattr(problem, "space_group", None),
        getattr(problem, "periodic_cell", None),
        getattr(problem, "grid", None),
        getattr(problem, "monkhorst_pack_dimensions", None),
        getattr(problem, "monkhorst_pack_shift", None),
    )


KPOINT_TIME_REVERSAL_ENV = "PARSEC_KPOINT_TIME_REVERSAL"


def kpoint_time_reversal_setting() -> str:
    """``PARSEC_KPOINT_TIME_REVERSAL``: ``auto`` (default, since 2026-11-18) reduces the whole Monkhorst-Pack mesh that
    forces and moving atoms keep (scalar or collinear-spin orbitals, the default routines of a bulk cell) to its
    pairs ``k``, ``-k`` (:func:`time_reversal_reduced`); ``off`` keeps the whole mesh, as before."""

    import os

    value = os.environ.get(KPOINT_TIME_REVERSAL_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{KPOINT_TIME_REVERSAL_ENV} must be auto or off, got {value!r}")
    return value


MP_UNSHIFTED_MESH_ENV = "PARSEC_MP_UNSHIFTED_MESH"


def unshifted_mesh_setting() -> str:
    """``PARSEC_MP_UNSHIFTED_MESH``: the mesh of a ``Monkhorst_Pack_Grid`` without a ``Monkhorst_Pack_Shift`` block.

    ``fortran`` (default, since 2026-11-18): PARSEC 1.5's, the Gamma-centred grid ``k = (i-1)/n`` reduced exactly
    as a zero ``Monkhorst_Pack_Shift`` is, without choosing the Fortran-derived routines as the block does.  On the
    Fortran-derived routines that is ``irrbz``'s wedge (the space group, and time reversal unless non-collinear).  On
    the default routines a single point, forces or a relaxation of scalar orbitals and of non-magnetic spin-orbit
    coupling (``PARSEC_SOC_KPOINT_WEDGE``) take the irreducible wedge of the operations that map the grid and the mesh
    onto themselves and keep the grid's finite-difference operators (:func:`stencil_preserving_operations`;
    ``PARSEC_SCALAR_KPOINT_WEDGE``), with time reversal, the density symmetrized (``Symmetrize_Charge_Density``
    true by default);
    a hybrid functional takes the time-reversal pairs of the mesh (its exchange sums over the whole mesh), molecular
    dynamics and manual movements the whole mesh as its time-reversal pairs, and non-collinear magnetism and
    spin-polarized spin-orbit coupling the whole mesh.  ``port``: this port's former unreduced grid
    ``(2i-n-1)/(2n)`` of equal weights (:func:`monkhorst_pack_grid`), bit for bit.  A mesh of one point
    (``1 1 1``) is the Gamma point under both.  The parser's note says which of these a run takes."""

    import os

    value = os.environ.get(MP_UNSHIFTED_MESH_ENV, "fortran").strip().lower() or "fortran"
    if value not in ("fortran", "port"):
        raise ValueError(f"{MP_UNSHIFTED_MESH_ENV} must be fortran or port, got {value!r}")
    return value


KPOINT_FORCE_WEDGE_ENV = "PARSEC_KPOINT_FORCE_WEDGE"


def kpoint_force_wedge_setting() -> str:
    """``PARSEC_KPOINT_FORCE_WEDGE``: ``auto`` (default, since 2026-11-18) lets the forces and the relaxation of a
    bulk cell on the default routines run the irreducible wedge of a Monkhorst-Pack mesh, as Fortran's routines and a
    single point do: the space group that maps the grid onto itself reduces the mesh and symmetrizes the density, the
    nonlocal forces of the wedge are completed over it and the forces averaged over it (``symm_forces.f90``), so the
    structure keeps the group; ``off`` keeps the whole mesh (its time-reversal pairs), as before.  Molecular dynamics
    and ``manual`` movements keep the whole mesh either way (PARSEC uses no symmetry for them)."""

    import os

    value = os.environ.get(KPOINT_FORCE_WEDGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{KPOINT_FORCE_WEDGE_ENV} must be auto or off, got {value!r}")
    return value


MOVE_FLAG_SPACE_GROUP_ENV = "PARSEC_MOVE_FLAG_SPACE_GROUP"


def move_flag_space_group_setting() -> str:
    """``PARSEC_MOVE_FLAG_SPACE_GROUP``: ``auto`` (default, since 2026-11-21) or ``off``.

    A relaxation by an optimizer (``bfgs``, ``simple``) of a bulk cell whose ``Move_Flag`` fixes atoms that an
    operation of the space group the problem carries (the irreducible wedge of a Monkhorst-Pack mesh, the density
    average of ``Symmetrize_Charge_Density``, a weighted ``Kpoint_List``) exchanges with movable ones leaves that group
    after its first move, but not the subgroup of the operations that map the fixed atoms onto fixed atoms
    (:func:`parsec_python.Grid.space_group.flag_preserving_subgroup`).  ``auto`` (with
    ``PARSEC_MOVE_FLAG_SUBGROUP=auto``) carries that subgroup: the mesh is reduced by it and time reversal, the density
    averaged over it, the nonlocal forces of the wedge completed and the forces averaged over it, on both routine
    sets.  ``off`` keeps the routes of before: the whole mesh as its time-reversal pairs on the default routines, a
    refusal by name on the Fortran-derived ones (and their Gamma-point density average over the whole group)."""

    import os

    value = os.environ.get(MOVE_FLAG_SPACE_GROUP_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{MOVE_FLAG_SPACE_GROUP_ENV} must be auto or off, got {value!r}")
    return value


SPINOR_MOVE_FLAG_SUBGROUP_ENV = "PARSEC_SPINOR_MOVE_FLAG_SUBGROUP"


def spinor_move_flag_subgroup_setting() -> str:
    """``PARSEC_SPINOR_MOVE_FLAG_SUBGROUP``: ``auto`` (default, since 2026-11-22) or ``off``.

    The subgroup of ``PARSEC_MOVE_FLAG_SPACE_GROUP`` (the operations that map the atoms ``Move_Flag`` fixes onto fixed
    atoms) for the spinors of non-magnetic self-consistent spin-orbit coupling, on both routine sets: their k-point
    mesh is reduced by the subgroup and time reversal (the SU(2) matrices turn a spinor, time reversal maps it onto
    its Kramers partner at ``-k``, neither changes the density or the forces), the density averaged over it, the
    forces completed and averaged over it, as for scalar orbitals.  ``off`` keeps the routes of before: the whole
    mesh as its time-reversal pairs on the default routines, the refusal by name on the Fortran-derived ones.
    Non-collinear magnetism and spin-polarized spin-orbit coupling keep their routes under both values (time
    reversal flips the moments), and so does a Gamma-point relaxation of spinors with ``Symmetrize_Charge_Density``
    on the Fortran-derived routines (its density is averaged over nothing, as before)."""

    import os

    value = os.environ.get(SPINOR_MOVE_FLAG_SUBGROUP_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{SPINOR_MOVE_FLAG_SUBGROUP_ENV} must be auto or off, got {value!r}")
    return value


SOC_KPOINT_WEDGE_ENV = "PARSEC_SOC_KPOINT_WEDGE"


def soc_kpoint_wedge_setting() -> str:
    """``PARSEC_SOC_KPOINT_WEDGE``: the Monkhorst-Pack mesh of non-magnetic spin-orbit coupling on the default routines.

    ``auto`` (default, since 2026-11-19): a bulk cell with self-consistent spin-orbit coupling, neither spin-polarized
    nor non-collinear, reduces its mesh as PARSEC 1.5 does (``irrbz`` with time reversal for every ``is_so`` run that
    is not ``ncl``): the irreducible wedge of the operations of the crystal that map the grid, the mesh and the
    finite-difference operators of the grid onto themselves (:func:`stencil_preserving_operations`), with time
    reversal, the scalar density averaged over them (a spinor turned with its operation and its SU(2) matrix has the
    turned density, and the Kramers partner at ``-k`` the same density), so that the wedge gives the energy of the
    same points without the group to SCF precision on any cell; the forces and a relaxation take the same wedge where
    ``PARSEC_KPOINT_FORCE_WEDGE`` lets scalar orbitals take it, and the time-reversal pairs of the whole mesh
    (``PARSEC_KPOINT_TIME_REVERSAL``) where they keep the whole mesh.  On an orthogonal grid that is every operation
    that maps the grid and the mesh onto themselves; on an oblique one (fcc, hcp) a subgroup.  ``grid``: the
    operations that map the grid and the mesh onto themselves, the group scalar orbitals took before 2026-11-21 and
    take under ``PARSEC_SCALAR_KPOINT_WEDGE=grid`` (fewer points on an oblique grid, at a discretization error: fcc
    Au 1.2e-4 Ry, hcp Au 7.7e-3 Ry at 0.3 bohr).  ``off`` keeps the
    whole mesh of equal weights, as before, bit for bit.  Non-collinear magnetism and collinear spin with spin-orbit
    coupling keep the whole mesh under every value (the moments are not part of the group found from the atoms, and
    time reversal flips them)."""

    import os

    value = os.environ.get(SOC_KPOINT_WEDGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "grid", "off"):
        raise ValueError(f"{SOC_KPOINT_WEDGE_ENV} must be auto, grid or off, got {value!r}")
    return value


SCALAR_KPOINT_WEDGE_ENV = "PARSEC_SCALAR_KPOINT_WEDGE"
#: The value of PARSEC_SCALAR_KPOINT_WEDGE without the variable.
SCALAR_KPOINT_WEDGE_DEFAULT = "auto"


def scalar_kpoint_wedge_setting() -> str:
    """``PARSEC_SCALAR_KPOINT_WEDGE``: the operations of the irreducible wedge of a Monkhorst-Pack mesh of scalar (and
    collinear-spin) orbitals of a bulk cell on the default routines (a single point, forces and a relaxation;
    :func:`parsec_python.Input.parsec_input._default_routines_wedge`).

    * ``auto`` (default, since 2026-11-21): those of the crystal that map the grid, the mesh and the grid's
      finite-difference operators onto themselves (:func:`stencil_preserving_operations`), with time reversal, the
      density averaged over them, as non-magnetic spin-orbit coupling takes them (``PARSEC_SOC_KPOINT_WEDGE=auto``);
      the forces over the larger group of :func:`kpoint_force_group_setting`.  With ``PARSEC_KPOINT_FOLD=shortest``
      the wedge then computes exactly the points of the whole mesh without symmetry on the same grid, its ties on the
      zone boundary shared (:func:`tie_shared_points`; A100: within 1.4e-10 Ry for fcc Au, hcp Au2, primitive bcc Au
      and primitive Si).  On an orthogonal grid that is every operation that maps the grid and the mesh onto
      themselves, bit for bit as ``grid``; on an oblique one (hexagonal, the primitive fcc and bcc cells) a subgroup
      (hcp 8 of 24, primitive fcc 12 of 48), so more k-points (1.75 to 2.4 times on the meshes measured, 1.3 to 1.8
      times the time).
    * ``grid``: every operation that maps the grid and the mesh onto themselves, as before: also the three- and
      six-fold axes of an oblique grid, which turn a grid axis onto another direction and change the
      finite-difference operators at the order of the discretization (A100, against the whole mesh of shared ties,
      0.3 bohr and order 6: hcp Au2 4x4x2 -1.2e-3 Ry, fcc Au 4x4x4, 6x6x6 and 8x8x8 -4.7e-4, -3.1e-4 and -9.7e-6 Ry,
      primitive bcc Au 6x6x6 +2.8e-3 Ry; order 12: hcp Au2 -7.2e-5 Ry, fcc Au 6x6x6 -2.5e-5 Ry).

    Any other value is refused before the SCF."""

    import os

    value = os.environ.get(SCALAR_KPOINT_WEDGE_ENV, SCALAR_KPOINT_WEDGE_DEFAULT).strip().lower()
    value = value or SCALAR_KPOINT_WEDGE_DEFAULT
    if value not in ("auto", "grid"):
        raise ValueError(f"{SCALAR_KPOINT_WEDGE_ENV} must be auto or grid, got {value!r}")
    return value


KPOINT_FOLD_ENV = "PARSEC_KPOINT_FOLD"
#: The value of PARSEC_KPOINT_FOLD without the variable.
KPOINT_FOLD_DEFAULT = "shortest"


def kpoint_fold_setting() -> str:
    """``PARSEC_KPOINT_FOLD``: the representatives of the k-points of a Monkhorst-Pack mesh on the default routines
    (a bulk cell, and a slab of ``Use_Fortran_Routines: .false.``).

    * ``shortest`` (default, since 2026-11-21): every point of the explicit list of the whole mesh (the mesh of
      forces and moving atoms without a wedge, of molecular dynamics, of a hybrid, of magnetic spinors, run whole or
      as its time-reversal pairs) and every representative of an irreducible wedge is the shortest of its images
      modulo the reciprocal lattice (:func:`fold_to_shortest`, the first Brillouin zone).  A point of the zone
      boundary with several shortest images runs as all of them with equal shares of its weight
      (``SinglePointInput.kpoint_tie_images``: :func:`tie_shared_points` in the whole mesh, the stars of a wedge split
      into orbits of exact images), except for a hybrid, which keeps one of them per point, the set closed under
      ``k -> -k`` (:func:`negation_closed_shortest`); the time-reversal pairs are then exact.  On a lattice of
      orthogonal vectors that is the point folded into ``(-1/2, 1/2]``, bit for bit as ``centred``.
    * ``centred``: every point folded into ``(-1/2, 1/2]`` along the reciprocal lattice vectors
      (:func:`fold_to_centered_zone`), as before; on an oblique cell some of them lie outside the first zone.

    The finite-difference Bloch operator of the u gauge is not invariant under ``k -> k + G``, so the choice changes
    the energy of an oblique cell at the order of the discretization.  The Fortran-derived routines keep the folded
    points under both values (PARSEC 1.5's ``irrbz`` keeps them in ``[0, 1)``), except a mesh with a free level below
    zero on routines the input did not request (:func:`kpoint_mesh_guard_setting`).  Any other value is refused before
    the SCF."""

    import os

    value = os.environ.get(KPOINT_FOLD_ENV, KPOINT_FOLD_DEFAULT).strip().lower() or KPOINT_FOLD_DEFAULT
    if value not in ("shortest", "centred"):
        raise ValueError(f"{KPOINT_FOLD_ENV} must be shortest or centred, got {value!r}")
    return value


KPOINT_LIST_FOLD_ENV = "PARSEC_KPOINT_LIST_FOLD"
#: The value of PARSEC_KPOINT_LIST_FOLD without the variable.
KPOINT_LIST_FOLD_DEFAULT = "auto"


def kpoint_list_fold_setting() -> str:
    """``PARSEC_KPOINT_LIST_FOLD``: the points of a ``Kpoint_List`` on an oblique cell.

    The discrete Bloch operator ``-L - 2i k.G + k^2`` of an oblique grid has plane waves with ``-L < |G|^2``, so it is
    not bounded below in ``k``: beyond some ``|k|``, in general outside the first Brillouin zone, it has levels below
    zero that no continuum state has (:func:`lowest_free_levels`).  On an orthogonal grid ``-L >= |G|^2``; its points
    outside the first zone follow :func:`kpoint_orthogonal_list_fold_setting` (``literal`` keeps them as written).

    * ``auto`` (default): on the default routines every point runs at its shortest images modulo the reciprocal
      lattice, a point of the zone boundary at all of them with equal shares of its weight (:func:`tie_shared_points`).
      Ties are found up to the precision of the input (:data:`WRITTEN_TIE_TOLERANCE`, :func:`written_tie_images`), so a
      whole mesh gives the result of its ``Monkhorst_Pack_Grid`` for any writing of its images; what stays is the
      rounding of the written coordinates themselves (a mesh with 8, 6 and 4 decimals: about 3e-9, 3e-7 and 3e-5 Ry on
      Au).  A hybrid keeps one image per point, those of the mesh whatever the order of the lines
      (:func:`negation_closed_written`), and ``Create_DOS`` and ``Output_GW`` keep one point per mesh point.  The
      Fortran-derived routines keep the written vectors, as PARSEC 1.5, unless a written point has a free level below
      zero: where they run without being asked for, that list is then treated as on the default routines; under
      ``Use_Fortran_Routines: .true.`` it keeps its vectors with a warning.
    * ``shared``: the treatment of the default routines on every route, the Fortran-derived routines included.
    * ``literal``: the written vectors on every route, bit for bit as before, with the same warning.

    A first-zone point can have such a level too on a very coarse grid (a bcc cell of 5.5 bohr with 6 points per axis):
    no image removes it, and the run goes on with a warning that the grid is too coarse.  Any other value is refused
    before the SCF."""

    import os

    value = os.environ.get(KPOINT_LIST_FOLD_ENV, KPOINT_LIST_FOLD_DEFAULT).strip().lower() or KPOINT_LIST_FOLD_DEFAULT
    if value not in ("auto", "shared", "literal"):
        raise ValueError(f"{KPOINT_LIST_FOLD_ENV} must be auto, shared or literal, got {value!r}")
    return value


KPOINT_ORTHOGONAL_LIST_FOLD_ENV = "PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD"


def kpoint_orthogonal_list_fold_setting() -> str:
    """``PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD``: the points of a ``Kpoint_List`` on an orthogonal cell that lie outside the
    first Brillouin zone.

    * ``auto`` (default): on the default routines and on Fortran-derived routines the run chose itself (and on every
      route under ``PARSEC_KPOINT_LIST_FOLD=shared``) such a point runs at its shortest image modulo the reciprocal
      lattice, with its weight, the same point of the zone.
      An orthogonal grid has no level below zero (:func:`kpoint_list_fold_setting`), but the finite-difference
      dispersion still departs from ``|k + G|^2`` the more the farther the written vector lies from the zone: a Bloch
      state of the zone point ``k_s`` written at ``k_s + G0`` is held by the waves ``q - G0``, nearer the edge of the
      grid (the plane waves within 2 bohr^-1 of a 7.71 bohr cube at 0.35 bohr and order 6 take a free-level error of
      3.6e-3 Ry at (3/4, 3/4, 3/4) against 3.1e-4 Ry at its image (-1/4, -1/4, -1/4)).  A point of the zone boundary,
      where its images are equally short (a component of 1/2), keeps its written vector: its images give the same
      levels and density by time reversal on these grids (the u-gauge operator at ``-k`` is the conjugate of that at
      ``k``); a list whose points all lie in the first zone keeps every bit.  ``Use_Fortran_Routines: .true.`` keeps the
      written vectors, as PARSEC 1.5, under ``PARSEC_KPOINT_LIST_FOLD=auto``, and ``literal`` keeps them everywhere.
    * ``off``: the written vectors of an orthogonal cell on every route, as before.

    Any other value is refused before the SCF."""

    import os

    value = os.environ.get(KPOINT_ORTHOGONAL_LIST_FOLD_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{KPOINT_ORTHOGONAL_LIST_FOLD_ENV} must be auto or off, got {value!r}")
    return value


KPOINT_MESH_GUARD_ENV = "PARSEC_KPOINT_MESH_GUARD"


def kpoint_mesh_guard_setting() -> str:
    """``PARSEC_KPOINT_MESH_GUARD``: a Monkhorst-Pack mesh with a point whose free level is below zero
    (:func:`lowest_free_levels`) on Fortran-derived routines that the input did not request.

    * ``auto`` (default): the mesh runs at the shortest representatives of its wedge, and outside a hybrid with the
      images of a zone-boundary point shared (``SinglePointInput.kpoint_shortest_representatives`` and
      ``kpoint_tie_images``; ``Create_DOS`` and ``Output_GW`` keep their points), with a note.  A mesh without such a
      point changes in nothing.
    * ``off``: the points of the mesh as before, with a warning.

    ``Use_Fortran_Routines: .true.`` keeps the mesh and the warning under both values.  Any other value is refused
    before the SCF."""

    import os

    value = os.environ.get(KPOINT_MESH_GUARD_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{KPOINT_MESH_GUARD_ENV} must be auto or off, got {value!r}")
    return value


KPOINT_GUARD_WEDGE_ENV = "PARSEC_KPOINT_GUARD_WEDGE"


def kpoint_guard_wedge_setting() -> str:
    """``PARSEC_KPOINT_GUARD_WEDGE``: the operations of the wedge of a mesh that :func:`kpoint_mesh_guard_setting`
    moves to the shortest representatives of its points, and the group its density is averaged over (a hybrid keeps
    its k-points: on its time-reversal pairs, the whole mesh, it averages no density, and its wedge averages over
    these operations, :func:`hybrid_wedge_operations`).

    * ``auto`` (default): the operations of the space group that also map the finite-difference operators of the grid
      onto themselves (:func:`stencil_preserving_operations` with :func:`grid_stencil_lines`: on Fortran's general
      grid of an oblique cell the axes and the auxiliary directions of its Laplacian, with the first differences along
      the axes), with time reversal, the density averaged over them, the forces over every operation of the space
      group under ``PARSEC_KPOINT_FORCE_GROUP=crystal`` (``SinglePointInput.force_space_group``).  The others turn a
      grid axis onto a direction with no first difference and change the discrete Bloch Hamiltonian at the order of
      the discretization, so that their wedge is not the energy of the same points without symmetry; with these the
      guarded mesh is the whole mesh at its shortest images with its ties shared, as the default routines take it
      (``PARSEC_SCALAR_KPOINT_WEDGE`` and ``PARSEC_SOC_KPOINT_WEDGE`` at ``auto``).  Where they are all the space group
      nothing changes.
    * ``crystal``: every operation of the space group, as before.

    Any other value is refused before the SCF."""

    import os

    value = os.environ.get(KPOINT_GUARD_WEDGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "crystal"):
        raise ValueError(f"{KPOINT_GUARD_WEDGE_ENV} must be auto or crystal, got {value!r}")
    return value


#: A free level counts as below zero under this value (Ry); exact levels are ``>= 0`` to round-off.
FREE_LEVEL_TOLERANCE = 1.0e-8


def _free_level_stencil(cell, settings):
    """``(counts, lines, steps, periodic axes)`` of the operators of the grid of ``settings``, or ``None`` for a wire.

    ``lines`` are the second differences ``(offset, weight)`` (weights in 1/bohr^2), ``steps`` the rows of the grid
    step vectors, without building the grid: :func:`grid_stencil_lines` of a bulk cell; the in-plane counts of a slab,
    whose confined axis is orthogonal to the plane and adds a non-negative term at no ``k``."""

    lattice = np.asarray(cell.lattice_vectors, dtype=np.float64)
    fortran = bool(getattr(settings, "fortran_routines", False))
    confined = int(getattr(settings, "periodicity", 3))
    if confined == 1 or int(cell.dimensions) == 1:
        return None
    if confined == 2:
        from ..Laplacian.nonorthogonal import pbc_grid_coefficients
        from .pbc import adjusted_steps

        lengths = np.sqrt(np.sum(lattice**2, axis=1))
        translations = getattr(settings, "fractional_translations", ()) or ((), (), ())
        points, spacings = adjusted_steps(lengths[:2], settings.spacing, translations[:2])
        spacings = np.asarray([*spacings, settings.spacing], dtype=np.float64)
        coefficients = pbc_grid_coefficients(lattice, spacings)
        lines = [
            (tuple(int(v) for v in offset), float(value) / float(step) ** 2)
            for offset, value, step in coefficients.stencil_terms(spacings)
        ]
        return np.asarray([*points, 1], dtype=np.int64), lines, (lattice / lengths[:, None]) * spacings[:, None], 2
    if int(cell.dimensions) == 2:
        in_plane = (2 * np.round(np.linalg.norm(lattice[:2], axis=1) / (2.0 * settings.spacing))).astype(np.int64)
        counts = np.asarray([in_plane[0], in_plane[1], int(lattice[2, 2] / settings.spacing) + 2], dtype=np.int64)
        return counts, default_stencil_lines(lattice, counts), lattice / counts[:, None], 2
    counts, lines = grid_stencil_lines(cell, settings)
    counts = np.asarray(counts, dtype=np.int64)
    if fortran:
        from .pbc import fortran_grid_counts

        _, spacings = fortran_grid_counts(cell, settings)
        lengths = np.sqrt(np.sum(lattice**2, axis=1))
        return counts, lines, (lattice / lengths[:, None]) * np.asarray(spacings, dtype=np.float64)[:, None], 3
    return counts, lines, lattice / counts[:, None], 3


def _free_level_slices(counts, lines, inverse_steps, periodic, order):
    """The symbols ``(A, D)`` of ``-L`` and the Cartesian gradient, one plane of waves of the first axis at a time.

    Every wave of the grid is in a plane, or its negative is, which has the same ``A`` and ``-D``, so the planes
    ``0 ... n_1 // 2`` hold them all.  ``A`` has shape ``(n_2, n_3)``, ``D`` ``(3, n_2, n_3)``."""

    from ..Laplacian import first_derivative_coefficients, second_derivative_coefficients

    width = int(order) // 2
    second = np.asarray(second_derivative_coefficients(order), dtype=np.float64)
    first = np.asarray(first_derivative_coefficients(order), dtype=np.float64)
    angles = [
        2.0 * np.pi * np.arange(int(count)) / int(count) if axis < periodic else np.zeros(1)
        for axis, count in enumerate(counts)
    ]
    along = [
        2.0 * sum(first[width + j] * np.sin(j * values) for j in range(1, width + 1)) for values in angles
    ]

    def second_symbol(phase):
        # c_0 + 2 sum_j c_j cos(j phase), by Clenshaw's recurrence
        x = np.cos(phase)
        upper, lower = 0.0, 0.0
        for j in range(width, 0, -1):
            upper, lower = 2.0 * second[width + j] + 2.0 * x * upper - lower, upper
        return second[width] + x * upper - lower

    theta_2, theta_3 = np.meshgrid(angles[1], angles[2], indexing="ij")
    along_2, along_3 = np.meshgrid(along[1], along[2], indexing="ij")
    fixed = np.zeros_like(theta_2)
    moving = []
    for offset, weight in lines:
        if offset[0] == 0:
            fixed -= float(weight) * second_symbol(float(offset[1]) * theta_2 + float(offset[2]) * theta_3)
        else:
            moving.append((float(offset[0]), float(offset[1]) * theta_2 + float(offset[2]) * theta_3, float(weight)))
    rest = np.stack([inverse_steps[c, 1] * along_2 + inverse_steps[c, 2] * along_3 for c in range(3)])
    for plane in range(len(angles[0]) // 2 + 1):
        laplacian = fixed.copy()
        for factor, phase, weight in moving:
            laplacian -= weight * second_symbol(factor * angles[0][plane] + phase)
        gradient = rest + (inverse_steps[:, 0] * along[0][plane])[:, None, None]
        yield laplacian, gradient


@lru_cache(maxsize=8)
def _free_level_reach(counts, lines, steps, periodic, order) -> float:
    """The smallest ``|k|`` at which a wave of the grid has a level of zero (``inf``: none has at any ``k``)."""

    inverse = np.linalg.inv(np.asarray(steps, dtype=np.float64).reshape(3, 3))
    reach = math.inf
    for laplacian, gradient in _free_level_slices(counts, lines, inverse, periodic, order):
        squares = np.sum(gradient**2, axis=0)
        unbounded = squares > laplacian + 1.0e-10
        if np.any(unbounded):
            size = squares[unbounded]
            reach = min(reach, float(np.min(np.sqrt(size) - np.sqrt(size - laplacian[unbounded]))))
    return reach


def lowest_free_levels(k_points, cell, settings) -> np.ndarray | None:
    """The lowest level of the free discrete Bloch operator ``-L - 2i k.G + k^2`` at each of ``k_points`` (Cartesian)
    on the grid of ``settings`` where that level is below zero.  A value that is not below zero (``+inf`` for a point
    too short to reach one) says only that no level is.  ``None`` where no point can have one (an orthogonal lattice
    or a wire).

    The operator is translation invariant, so the plane waves of the grid are its eigenvectors, with the level
    ``A + 2 k.D + k^2`` of the symbols ``A`` of ``-L`` and ``D`` of the Cartesian gradient.  A wave with ``A >= |D|^2``
    has a level ``>= |k + D|^2 >= 0`` at every ``k`` (on an orthogonal grid every wave: each second difference bounds
    the square of its first difference), and any other reaches zero first at ``|k| = |D| - sqrt(|D|^2 - A)``.  Every
    wave of the grid is visited, a plane at a time (:func:`_free_level_slices`, no more memory than one plane): once
    for the smallest such ``|k|`` of the grid, and again only if a point lies beyond it.  The result depends on the
    cell, the settings and the point alone, so every MPI rank gets the same.
    """

    points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    if not oblique_reciprocal_lattice(reciprocal_lattice(cell.lattice_vectors)):
        return None
    stencil = _free_level_stencil(cell, settings)
    if stencil is None:
        return None
    counts, lines, steps, periodic = stencil
    key_counts = tuple(int(value) for value in counts)
    key_lines = tuple((tuple(int(v) for v in offset), float(weight)) for offset, weight in lines)
    key_steps = tuple(float(value) for value in np.asarray(steps, dtype=np.float64).ravel())
    order = int(settings.expansion_order)
    reach = _free_level_reach(key_counts, key_lines, key_steps, int(periodic), order)
    levels = np.full(points.shape[0], np.inf)
    if not math.isfinite(reach):
        return levels
    lengths = np.sum(points**2, axis=1)
    beyond = np.flatnonzero(lengths >= reach**2)
    if beyond.size == 0:
        return levels
    inverse = np.linalg.inv(np.asarray(steps, dtype=np.float64))
    chosen = points[beyond]
    squared = lengths[beyond]
    best = np.full(beyond.size, np.inf)
    for laplacian, gradient in _free_level_slices(key_counts, key_lines, inverse, int(periodic), order):
        unbounded = np.sum(gradient**2, axis=0) > laplacian + 1.0e-10
        if not np.any(unbounded):
            continue
        waves = laplacian[unbounded]
        vectors = gradient[:, unbounded]
        for start in range(0, beyond.size, 64):
            part = np.arange(start, min(start + 64, beyond.size))
            # (the wave of -theta has the same A and -D)
            values = waves[None, :] + squared[part, None] - 2.0 * np.abs(chosen[part] @ vectors)
            best[part] = np.minimum(best[part], np.min(values, axis=1))
    levels[beyond] = best
    return levels


def shortest_list_points(k_points, lattice_vectors, *, tolerance: float = SHORTEST_FOLD_TOLERANCE):
    """``(points, moved)`` of Cartesian ``k_points``: each point longer than the shortest of its images modulo the
    reciprocal lattice of ``lattice_vectors`` replaced by that image (:func:`fold_to_shortest`), ``moved`` flagging
    them.  A point within ``tolerance`` (relative to the largest ``|b_i|^2``) of the shortest length keeps its bits."""

    k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    reciprocal = reciprocal_lattice(lattice_vectors)
    shortest = np.atleast_2d(fold_to_shortest(k_points @ np.linalg.inv(reciprocal), reciprocal, tolerance=tolerance))
    shortest = shortest @ reciprocal
    scale = float(np.max(np.sum(reciprocal**2, axis=1)))
    moved = np.sum(k_points**2, axis=1) > np.sum(shortest**2, axis=1) + tolerance * scale
    points = k_points.copy()
    points[moved] = shortest[moved]
    return points, moved


KPOINT_FORCE_GROUP_ENV = "PARSEC_KPOINT_FORCE_GROUP"
#: The value of PARSEC_KPOINT_FORCE_GROUP without the variable.
KPOINT_FORCE_GROUP_DEFAULT = "crystal"


def kpoint_force_group_setting() -> str:
    """``PARSEC_KPOINT_FORCE_GROUP``: the operations the forces and the moved positions of a wedge of the default
    routines are averaged over where the wedge keeps only those that keep the grid's finite-difference operators
    (``PARSEC_SCALAR_KPOINT_WEDGE=auto`` or ``PARSEC_SOC_KPOINT_WEDGE=auto`` on an oblique grid).

    * ``crystal`` (default, since 2026-11-21): every operation of the space group that maps the grid and the mesh onto
      themselves (``SinglePointInput.force_space_group``), the group of the forces before 2026-11-21, while the k-point
      sum stays that of the wedge.  The forces of the converged grid have that symmetry; the average is the orthogonal
      projection onto it, so it removes only the part the oblique grid breaks (an hcp atom on the three-fold axis
      keeps zero in-plane forces) and keeps a relaxed structure in the group.
    * ``wedge``: the operations of the wedge alone (the forces of the same points without symmetry on that grid).

    Any other value is refused before the SCF."""

    import os

    value = os.environ.get(KPOINT_FORCE_GROUP_ENV, KPOINT_FORCE_GROUP_DEFAULT).strip().lower()
    value = value or KPOINT_FORCE_GROUP_DEFAULT
    if value not in ("crystal", "wedge"):
        raise ValueError(f"{KPOINT_FORCE_GROUP_ENV} must be crystal or wedge, got {value!r}")
    return value


#: The switch of the forces of a ``Kpoint_List`` of the default routines (:func:`kpoint_list_force_group_setting`).
KPOINT_LIST_FORCE_GROUP_ENV = "PARSEC_KPOINT_LIST_FORCE_GROUP"


def kpoint_list_force_group_setting() -> str:
    """``PARSEC_KPOINT_LIST_FORCE_GROUP``: the operations the forces and the moved positions of a ``Kpoint_List`` of
    scalar or collinear-spin orbitals on the default routines of a bulk cell are averaged over.

    * ``auto`` (default, since 2026-11-22): the operations of the space group that map the grid and the weighted list
      (its classes ``{k, -k}`` modulo the reciprocal lattice) onto themselves
      (:func:`list_preserving_operations`; ``SinglePointInput.force_space_group``), as the forces of the same points
      given as a ``Monkhorst_Pack_Grid`` are (``PARSEC_KPOINT_FORCE_GROUP=crystal``): the forces of the converged grid
      have that symmetry, and the average removes only the part the oblique grid breaks (the in-plane forces of an
      hcp atom on the three-fold axis, 4.2e-5 Ry/bohr for Au2 at 0.3 bohr).  Nothing where only the identity keeps
      the list, with ``Ignore_Symmetry``, or where an operation exchanges fixed and movable atoms.
    * ``off``: no average, as before.

    Any other value is refused before the SCF."""

    import os

    value = os.environ.get(KPOINT_LIST_FORCE_GROUP_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{KPOINT_LIST_FORCE_GROUP_ENV} must be auto or off, got {value!r}")
    return value


def time_reversal_reduced(k_points, weights, lattice_vectors) -> tuple[np.ndarray, np.ndarray]:
    """The representatives of a uniform mesh under time reversal (``k ~ -k``).

    The first point of each pair is kept with the weights of both; a point with
    ``-k = k`` modulo the reciprocal lattice of ``lattice_vectors`` keeps its
    own.  The order of the first members is kept.  The partner is ``-k``
    itself where the set holds it, else ``-k`` plus a reciprocal vector (the
    only partner of a set with one point per class, as before 2026-11-21; a
    whole mesh of shared ties, :func:`tie_shared_points`, holds both).
    """

    k_points = np.atleast_2d(np.asarray(k_points, dtype=float))
    weights = np.asarray(weights, dtype=float)
    fractions = k_points @ np.linalg.inv(reciprocal_lattice(lattice_vectors))
    taken = np.zeros(k_points.shape[0], dtype=bool)
    kept, kept_weights = [], []
    for index in range(k_points.shape[0]):
        if taken[index]:
            continue
        taken[index] = True
        weight = float(weights[index])
        total = fractions + fractions[index]
        partners = np.flatnonzero(np.all(np.abs(total - np.round(total)) < 1.0e-8, axis=1) & ~taken)
        if partners.size:
            exact = partners[np.all(np.abs(total[partners]) < 1.0e-8, axis=1)]
            partner = int(exact[0]) if exact.size else int(partners[0])
            taken[partner] = True
            weight += float(weights[partner])
        kept.append(index)
        kept_weights.append(weight)
    return k_points[kept].copy(), np.asarray(kept_weights, dtype=float)


def kpoint_sampling(problem) -> tuple[np.ndarray, np.ndarray] | None:
    """``(k_points, weights)`` requested by a ``SinglePointInput``, or ``None``.

    An explicit list (``problem.kpoints``), Fortran's reduced grid
    (``problem.monkhorst_pack_shift`` given, :func:`kpoint_sampling_details`;
    also the zero shift of a mesh without a ``Monkhorst_Pack_Shift`` block)
    or this port's former unreduced Monkhorst-Pack grid
    (``problem.monkhorst_pack_dimensions`` alone: a mesh of one point, or
    ``PARSEC_MP_UNSHIFTED_MESH=port``); ``None`` means Gamma only.  For a
    hybrid functional the port's grid and a list of equal weights are reduced
    by time reversal (:func:`time_reversal_reduced`) and Fortran's grid by
    time reversal only; the exchange checks that the set is a uniform mesh
    (:mod:`parsec_python.SCF.hybrid_periodic`).
    """

    if getattr(problem, "kpoints", None) is not None:
        k_points = np.asarray(problem.kpoints, dtype=float)
        weights = np.asarray(problem.kpoint_weights, dtype=float)
        if getattr(problem, "kpoint_tie_images", False):
            # the whole mesh of the default routines on an oblique cell: every shortest image of a point on the zone
            # boundary with its share of the weight (exact time-reversal pairs)
            lattice = problem.periodic_cell.lattice_vectors
            k_points, weights = tie_shared_points(
                k_points, weights, lattice, written=bool(getattr(problem, "kpoint_written_ties", False))
            )
            if getattr(problem, "kpoint_time_reversal", False):
                return time_reversal_reduced(k_points, weights, lattice)
            return k_points, weights
        if (_hybrid(problem) or getattr(problem, "kpoint_time_reversal", False)) and np.allclose(
            weights, weights[0], rtol=1.0e-12, atol=0.0
        ):
            # a list of equal weights (a full mesh written out): its time-reversal representatives (a hybrid; the
            # mesh of the forces and moving atoms of scalar or collinear-spin orbitals or of non-magnetic spin-orbit
            # coupling, kpoint_time_reversal)
            return time_reversal_reduced(k_points, weights, problem.periodic_cell.lattice_vectors)
        return k_points, weights
    details = kpoint_sampling_details(problem)
    if details is not None:
        return details.k_points.copy(), details.weights.copy()
    if problem.monkhorst_pack_dimensions is not None:
        k_points, weights = monkhorst_pack_grid(problem.periodic_cell, problem.monkhorst_pack_dimensions)
        if _hybrid(problem):
            return time_reversal_reduced(k_points, weights, problem.periodic_cell.lattice_vectors)
        return k_points, weights
    return None


__all__ = [
    "IBZ_TOLERANCE",
    "KPOINT_UNITS",
    "MOVE_FLAG_SPACE_GROUP_ENV",
    "SPINOR_MOVE_FLAG_SUBGROUP_ENV",
    "MonkhorstPackSampling",
    "explicit_kpoints",
    "band_path_kpoints",
    "fold_to_centered_zone",
    "hybrid_kpoint_force_wedge_setting",
    "fold_to_shortest",
    "negation_closed_shortest",
    "oblique_reciprocal_lattice",
    "shortest_images",
    "tie_shared_points",
    "kpoint_fold_setting",
    "FREE_LEVEL_TOLERANCE",
    "WRITTEN_TIE_TOLERANCE",
    "kpoint_list_fold_setting",
    "kpoint_mesh_guard_setting",
    "kpoint_guard_wedge_setting",
    "kpoint_orthogonal_list_fold_setting",
    "lowest_free_levels",
    "negation_closed_written",
    "written_tie_images",
    "shortest_list_points",
    "kpoint_force_group_setting",
    "scalar_kpoint_wedge_setting",
    "hybrid_kpoint_wedge_setting",
    "DeferredHybridWedge",
    "HYBRID_WEDGE_FRACTIONAL_TOLERANCE",
    "default_stencil_lines",
    "grid_is_oblique",
    "grid_stencil_lines",
    "hybrid_wedge_group",
    "hybrid_wedge_operations",
    "irreducible_monkhorst_pack",
    "kpoint_sampling",
    "kpoint_force_wedge_setting",
    "kpoint_sampling_details",
    "kpoint_time_reversal_setting",
    "mesh_preserving_operations",
    "move_flag_space_group_setting",
    "spinor_move_flag_subgroup_setting",
    "monkhorst_pack_grid",
    "partly_filled_states",
    "read_kpoint_file",
    "soc_kpoint_wedge_setting",
    "stencil_preserving_matrices",
    "stencil_preserving_operations",
    "time_reversal_reduced",
    "translate01",
    "unshifted_mesh_setting",
]
