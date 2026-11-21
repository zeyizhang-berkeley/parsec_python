"""Which part of a point group the orbitals use: one-dimensional sectors of a subgroup, with weights.

The orbitals of a symmetric structure are solved in sectors: the functions that transform under a group ``H``
by a real one-dimensional character ``lambda`` (``psi(h r) = lambda(h) psi(r)``), each sector a basis on the
orbits of ``H`` (:mod:`.representations`).  For an abelian ``H`` of order two per element (the reflection groups
of PARSEC, D2h and its subgroups) these characters are all of its irreducible representations and every sector
state is one state.  The full point group ``G`` of a structure is often larger and not abelian: a nanodiamond
cut around a carbon atom has Td (24 operations), whose irreducible representations E, T1, T2 are two- and
three-dimensional.  Its subgroup D2 gives four sectors, three of which (B1, B2, B3) hold the same levels: the
three partners of every T level, exchanged by the threefold rotations.

This module chooses a subgroup ``H`` of ``G`` and the sectors to solve so that every level of ``G`` is found
and counted once with its degeneracy, at the least cost:

* the sectors are the real one-dimensional characters ``lambda`` of ``H`` (a non-abelian ``H`` has fewer of them
  than elements; its higher-dimensional representations are not solved);
* a level of the ``G``-representation ``Gamma`` (dimension ``d``) appears in sector ``lambda`` as often as
  ``lambda`` occurs in the restriction of ``Gamma`` to ``H`` (``m(lambda, Gamma)``), so ``n1(Gamma) =
  sum_lambda m(lambda, Gamma)`` times in all.  Every ``Gamma`` must occur (``n1 >= 1``), and a sector ``lambda``
  counts each of its states ``w_lambda = d / n1(Gamma)`` times, which must be the same for every ``Gamma`` that
  it holds (otherwise the subgroup is not admitted: a state's weight would depend on its unknown ``Gamma``);
* sectors that an element of the normalizer of ``H`` in ``G`` maps onto each other hold the same levels: one of
  them is solved, with the weight of all;
* the cost of a sector is its rows times its states, about ``N / |H|`` times ``M sum_Gamma m(lambda, Gamma)
  d / |G|`` (``N`` grid points, ``M`` states); the admitted subgroup of the least total cost is chosen.

For Td this is D2d (the S4 axis along a coordinate axis): sectors A1, B1, A2, B2 of about ``N/8`` rows and
``M/8`` states each, with weights 1, 1, 3, 3 (A2 and B2 hold one partner of every T1 and T2 level; the two
others lie in the E of D2d, which is the threefold image of these), a quarter of the basis of D2 (four sectors
of ``N/4`` by ``M/4``).  For Oh it is D4h (eight sectors of ``N/16`` by ``3M/48``, weights 1, 3, 1, 3, ...).
For an abelian reflection group it is the group itself, with weight 1 everywhere: the former sectors.

Everything here acts on the operations as 3 x 3 integer matrices (signed permutations), by their
multiplication table; the characters of ``G`` come from its class multiplication coefficients (Burnside's
method), and a pair of complex conjugate representations, whose levels real orbitals join, counts as one real
representation of twice the dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np


def _key(matrix: np.ndarray) -> bytes:
    return np.asarray(np.rint(matrix), dtype=np.int64).tobytes()


def _reduced(matrix: np.ndarray, modulus) -> np.ndarray:
    """An affine ``[[R, t], [0, 1]]`` with ``t`` taken modulo ``modulus`` (the translations of a periodic grid)."""

    if modulus is None:
        return matrix
    reduced = matrix.copy()
    reduced[:3, 3] %= np.asarray(modulus, dtype=np.int64)
    return reduced


#: products formed at once by :func:`multiplication_table` (entries of the ``(rows, order, d, d)`` block)
_PRODUCT_BLOCK_ENTRIES = 1 << 22


def multiplication_table(operations, modulus=None) -> np.ndarray:
    """``table[i, j]`` = index of ``operations[i] @ operations[j]``; raises unless the set is a group.

    ``modulus`` (points per axis): the operations are affine ``4 x 4`` matrices whose translations (grid steps)
    count modulo it, those of a periodic cell.

    The products are formed for blocks of rows at once and found among the operations by an integer code of their
    entries (a mixed radix over the range of every entry among the operations: a product with an entry outside it is
    no operation), the table of the loop over every pair of before, entry for entry, in a few milliseconds for the 384
    operations of a rock-salt 2x2x2 cell (a second before).
    """

    matrices = [_reduced(np.asarray(np.rint(op), dtype=np.int64), modulus) for op in operations]
    order = len(matrices)
    if order == 0:
        return np.empty((0, 0), dtype=np.int64)
    stack = np.stack(matrices)
    shape = stack.shape[1:]
    flat = stack.reshape(order, -1)
    if np.unique(flat, axis=0).shape[0] != order:
        raise ValueError("the operations are not distinct")
    low = flat.min(axis=0)
    span = flat.max(axis=0) - low + 1
    if float(np.prod(span.astype(np.float64))) >= 2.0**62:
        return _multiplication_table_by_pairs(matrices, modulus)
    radix = np.concatenate(([1], np.cumprod(span[:-1]))).astype(np.int64)
    codes = (flat - low) @ radix
    by_code = np.argsort(codes, kind="stable")
    sorted_codes = codes[by_code]
    reduce_rows = modulus is not None and len(shape) == 2 and shape[0] == 4
    counts = None if not reduce_rows else np.asarray(modulus, dtype=np.int64)
    table = np.empty((order, order), dtype=np.int64)
    rows = max(1, _PRODUCT_BLOCK_ENTRIES // max(1, order * flat.shape[1]))
    for start in range(0, order, rows):
        stop = min(order, start + rows)
        products = np.einsum("aij,bjk->abik", stack[start:stop], stack)
        if reduce_rows:
            products[:, :, :3, 3] %= counts
        entries = products.reshape(stop - start, order, -1)
        inside = np.all((entries >= low) & (entries < low + span), axis=2)
        product_codes = np.where(inside, (entries - low) @ radix, -1)
        position = np.searchsorted(sorted_codes, product_codes)
        position = np.minimum(position, order - 1)
        found = inside & (sorted_codes[position] == product_codes)
        if not np.all(found):
            raise ValueError("the operations are not closed under multiplication")
        table[start:stop] = by_code[position]
    return table


def _multiplication_table_by_pairs(matrices, modulus) -> np.ndarray:
    """:func:`multiplication_table` by the loop over every pair (entries whose codes would not fit 62 bits)."""

    index = {_key(m): i for i, m in enumerate(matrices)}
    order = len(matrices)
    table = np.empty((order, order), dtype=np.int64)
    for i, a in enumerate(matrices):
        for j, b in enumerate(matrices):
            k = index.get(_key(_reduced(a @ b, modulus)))
            if k is None:
                raise ValueError("the operations are not closed under multiplication")
            table[i, j] = k
    return table


def _identity_and_inverses(table: np.ndarray) -> tuple[int, np.ndarray]:
    order = table.shape[0]
    identity = int(np.flatnonzero(np.all(table == np.arange(order)[None, :], axis=1))[0])
    # the first (the only) j with g j = e in every row
    inverse = np.argmax(table == identity, axis=1).astype(np.int64)
    return identity, inverse


def conjugacy_classes(table: np.ndarray) -> list[tuple[int, ...]]:
    """The conjugacy classes (element indices), the class of the identity first."""

    identity, inverse = _identity_and_inverses(table)
    order = table.shape[0]
    seen = np.zeros(order, dtype=bool)
    classes = []
    every = np.arange(order)
    for g in [identity] + [i for i in range(order) if i != identity]:
        if seen[g]:
            continue
        # h g h^-1 for every h, as a sorted set
        members = np.unique(table[table[every, g], inverse]).tolist()
        seen[members] = True
        classes.append(tuple(int(m) for m in members))
    return classes


@dataclass(frozen=True)
class RealRepresentation:
    """A real irreducible representation of a group: its dimension and character per element."""

    dimension: int
    character: np.ndarray  # float, one value per element of the group
    # a pair of complex conjugate representations joined (a level of each and its conjugate are one real level)
    complex_pair: bool = False

    @property
    def levels(self) -> float:
        """Levels per ``M / |G|`` states: ``d`` partners each, or ``d/2`` real levels of a joined pair."""

        return self.dimension / 2.0 if self.complex_pair else float(self.dimension)


def real_irreducible_characters(table: np.ndarray) -> list[RealRepresentation]:
    """The real irreducible representations of the group of ``table``.

    Burnside's method: the class sums multiply as ``C_i C_j = sum_k a_ijk C_k``, and ``omega_k = |C_k|
    chi(C_k) / d`` of every irreducible character is a common eigenvector of the matrices ``(A_i)_jk =
    a_ijk`` (eigenvalue ``omega_i``).  A pair of complex conjugate characters becomes one real representation
    of twice the dimension (character ``2 Re chi``).  The characters are checked for orthogonality.
    """

    classes = conjugacy_classes(table)
    order = table.shape[0]
    count = len(classes)
    class_of = np.empty(order, dtype=np.int64)
    for c, members in enumerate(classes):
        class_of[list(members)] = c
    sizes = np.array([len(members) for members in classes], dtype=np.float64)
    coefficients = np.zeros((count, count, count))
    for i, ci in enumerate(classes):
        # the class of x y for x in C_i and every y, counted per (class of y, class of x y): integers, as the tally
        # of before
        products = class_of[table[np.asarray(ci, dtype=np.int64)]]
        pairs = class_of[None, :] * count + products
        tally = np.bincount(pairs.reshape(-1), minlength=count * count).reshape(count, count).astype(np.float64)
        coefficients[i] = tally / sizes
    generator = np.random.default_rng(20261115)
    combined = np.tensordot(generator.standard_normal(count), coefficients, axes=(0, 0))
    _values, vectors = np.linalg.eig(combined)
    complex_characters = []
    for column in range(count):
        omega = vectors[:, column] / vectors[0, column]
        dimension = np.sqrt(order / np.sum(np.abs(omega) ** 2 / sizes).real)
        d = int(round(float(dimension)))
        if d < 1 or abs(dimension - d) > 1e-6:
            raise RuntimeError("the class sums gave no integer dimension")
        complex_characters.append((d, d * omega / sizes))
    # orthogonality of the class characters
    gram = np.array([[np.sum(sizes * a * np.conj(b)) for _, b in complex_characters] for _, a in complex_characters])
    if not np.allclose(gram, order * np.eye(count), atol=1e-6):
        raise RuntimeError("the computed characters are not orthogonal")
    representations: list[RealRepresentation] = []
    used = [False] * count
    for a, (d, chi) in enumerate(complex_characters):
        if used[a]:
            continue
        used[a] = True
        pair = False
        if np.allclose(chi.imag, 0.0, atol=1e-8):
            values = chi.real
            dimension = d
        else:
            pair = True
            partner = next(
                b for b in range(count)
                if not used[b] and np.allclose(complex_characters[b][1], np.conj(chi), atol=1e-8)
            )
            used[partner] = True
            values = 2.0 * chi.real
            dimension = 2 * d
        representations.append(RealRepresentation(dimension, np.asarray(values[class_of], dtype=np.float64), pair))
    representations.sort(key=lambda rep: (rep.dimension, -float(np.sum(rep.character))))
    return representations


def _closure(table: np.ndarray, generators) -> frozenset[int]:
    identity, _ = _identity_and_inverses(table)
    members = {identity}
    frontier = [identity]
    generators = [int(g) for g in generators]
    while frontier:
        nxt = []
        for x in frontier:
            for g in generators:
                y = int(table[x, g])
                if y not in members:
                    members.add(y)
                    nxt.append(y)
        frontier = nxt
    return frozenset(members)


def _extended(table: np.ndarray, members: np.ndarray, element: int) -> frozenset[int]:
    """The subgroup generated by the subgroup ``members`` (a boolean mask) and ``element``."""

    mask = members.copy()
    generators = np.append(np.flatnonzero(members), element).astype(np.int64)
    frontier = np.array([element], dtype=np.int64)
    mask[element] = True
    while frontier.size:
        images = table[frontier][:, generators].reshape(-1)
        images = np.unique(images[~mask[images]])
        mask[images] = True
        frontier = images
    return frozenset(np.flatnonzero(mask).tolist())


def subgroups(table: np.ndarray, start=()) -> list[frozenset[int]]:
    """Every subgroup of the group of ``table`` that contains ``start`` (default: every subgroup): the one that
    ``start`` generates, extended by one element at a time to a fixed point.

    Every subgroup is reached: a subgroup ``K`` contains a chain ``{e} < <g_1> < <g_1, g_2> < ... = K``.  The
    extensions of ``S`` by ``g`` and by ``g s`` (``s`` in ``S``) are the same group, so one element of each left
    coset ``g S`` is tried.  The order is that of before: by size, then by the sorted members.  (A subgroup
    ``K`` that contains ``start`` contains such a chain from the group ``start`` generates.)
    """

    order = table.shape[0]
    trivial = _closure(table, tuple(int(g) for g in start))
    found = {trivial}
    frontier = {trivial}
    while frontier:
        new = set()
        for group in frontier:
            members = np.zeros(order, dtype=bool)
            members[list(group)] = True
            tried = members.copy()
            group_members = np.flatnonzero(members)
            for g in range(order):
                if tried[g]:
                    continue
                tried[table[g, group_members]] = True
                extended = _extended(table, members, g)
                if extended not in found:
                    new.add(extended)
        found |= new
        frontier = new
    return sorted(found, key=lambda s: (len(s), sorted(s)))


def one_dimensional_characters(table: np.ndarray, members) -> list[dict[int, int]]:
    """The real one-dimensional characters (homomorphisms to {+1, -1}) of the subgroup ``members``."""

    members = sorted(int(m) for m in members)
    identity, _ = _identity_and_inverses(table)
    generators: list[int] = []
    span = _closure(table, ())
    for g in members:
        if g not in span:
            generators.append(g)
            span = _closure(table, generators)
    count = len(generators)
    # The walk from the identity along the generators (breadth first, the generators in order), once: the order in
    # which every element is reached and the parity of every generator on its path, a bit mask (generator i is bit
    # count-1-i, so that the masks of the sign choices count in the order of np.ndindex).  The walk of a choice of
    # signs reaches the elements in this order and gives element y the sign (-1)^(popcount(path[y] & choice)); a
    # choice whose signs are a homomorphism (every product consistent) is a character, every other one is not.
    path = {identity: 0}
    reached = [identity]
    frontier = [identity]
    while frontier:
        nxt = []
        for x in frontier:
            for bit, g in enumerate(generators):
                y = int(table[x, g])
                if y not in path:
                    path[y] = path[x] ^ (1 << (count - 1 - bit))
                    reached.append(y)
                    nxt.append(y)
        frontier = nxt
    if len(path) != len(members) or set(path) != set(members):
        # (not a subgroup: no choice of signs gives values on exactly its elements, as before)
        return []
    order = table.shape[0]
    position = np.full(order, -1, dtype=np.int64)
    position[np.asarray(members, dtype=np.int64)] = np.arange(len(members))
    masks = np.zeros(len(members), dtype=np.int64)
    for element, mask in path.items():
        masks[position[element]] = mask
    products = position[table[np.ix_(members, members)]]
    # a choice is a homomorphism where it is even on every relation path[a b] ^ path[a] ^ path[b]
    relations = np.unique(masks[products] ^ masks[:, None] ^ masks[None, :])
    relations = relations[relations != 0]
    choices = np.arange(1 << count, dtype=np.int64)
    admitted = np.ones(choices.size, dtype=bool)
    for relation in relations.tolist():
        admitted &= _even_parity(choices & relation)
    characters = []
    for choice in np.flatnonzero(admitted).tolist():
        characters.append({y: (-1 if bin(path[y] & choice).count("1") % 2 else 1) for y in reached})
    return characters


def _even_parity(values: np.ndarray) -> np.ndarray:
    """Whether every value has an even number of set bits."""

    values = values.copy()
    parity = np.zeros(values.shape, dtype=np.int64)
    while np.any(values):
        parity ^= values & 1
        values >>= 1
    return parity == 0


@dataclass(frozen=True)
class IrrepUnit:
    """One solved unit of the irreducible-representation plan (:func:`irrep_units`).

    The states of the solved sector ``sector`` (an index into the plan's ``characters``: a one-dimensional
    character ``lambda`` of ``H``) that lie in the real irreducible representations ``representations`` of ``G``
    (indices into :func:`real_irreducible_characters` of the group table).  Each state counts ``weight`` times
    (``d / m(lambda, Gamma)``, the same for every representation of the unit); ``share`` is its expected share of
    the states (``sum m(lambda, Gamma) levels(Gamma) / |G|``, in units of ``M``).  ``projection``: ``None`` where
    the sector holds these representations alone (the sector of before, solved whole), else the coefficients
    ``b_c`` of the projector onto them on the sector, one per right coset representative ``c`` of the plan
    (``P = sum_c b_c T_c``, :meth:`.representations.ReflectionRepresentationDecomposition.projection_tables`):

    ``b_c = sum_Gamma (levels(Gamma) / |G|) sum_{h in H} chi_Gamma(h c) lambda(h)``.
    """

    sector: int
    representations: tuple[int, ...]
    weight: int
    share: float
    projection: tuple[float, ...] | None
    # (with the partner occupations of the weighted sectors) the representations of the unit whose levels it
    # holds in several states (``m(lambda, Gamma) > 1``): their partners take one occupation each, as
    # :attr:`OrbitalGroupPlan.partner_irreps` says of the sectors of before; a representation solved in one state
    # per level (``m = 1``, its other partners projected out of the other sectors) has none
    partners: tuple[int, ...] = ()
    # The positions (into the plan's ``cosets``, the identity first) of the right cosets ``H c`` that make a larger
    # subgroup ``K`` of ``G`` whose one-dimensional sector the projected unit is, whole: its projector is
    # ``(1/[K:H]) sum_c mu(c) T_c`` over those cosets (the coefficients ``+-1/[K:H]`` there and zero elsewhere).  The
    # unit's states are then the functions of that sector of ``K`` and can be solved on the orbits of ``K``
    # (``PARSEC_ORBITAL_IRREP_ROWS=partner``, :meth:`.representations.ReflectionRepresentationDecomposition.partner_rows`):
    # one partner row of each of its representations, the others the images under the cosets of ``K``.  ``None``:
    # a unit solved whole (no projector) or one that is no such sector (the E of Td in a sector of D2d).
    subgroup_cosets: tuple[int, ...] | None = None


@dataclass(frozen=True)
class OrbitalGroupPlan:
    """The orbital subgroup, its solved sectors and their weights, and the right cosets of the full group.

    ``subgroup``: element indices of ``H`` in the full group; ``characters``: one dict per solved sector
    (element index -> +1/-1 on ``H``); ``weights``: the number of states each state of that sector counts
    for; ``cosets``: indices of right coset representatives ``c`` (``G`` = union of ``H c``), the identity
    first; ``cost``: the solved rows times states in units of ``N M``; ``abelian_cost``: the same for the
    largest abelian subgroup of order two per element (the former route).  ``units``: the units of the
    irreducible representations of ``G`` that :func:`irrep_units` gives these sectors where the plan was asked
    for them (``units=`` of :func:`orbital_group_plan`), else ``None``.

    ``partner_irreps``: per solved sector, the indices (into the real irreducible representations of ``G``) of
    those it holds whose partners several solved states share: the level splits over sectors (the e of Td over
    the sectors A1 and B1 of D2d, weight 1 each) or over two states of one sector.  The states of such a level
    stay degenerate, each with an occupation of its own (``Occupations.partner_occupations`` checks that fixed
    occupations fill them alike).  Empty for the abelian route of before.  (With ``units`` the reduction takes
    those of the units instead, :attr:`IrrepUnit.partners`.)
    """

    subgroup: tuple[int, ...]
    characters: tuple[dict, ...]
    weights: tuple[int, ...]
    cosets: tuple[int, ...]
    cost: float
    abelian_cost: float
    partner_irreps: tuple[tuple[int, ...], ...] = ()
    units: tuple[IrrepUnit, ...] | None = None
    # the rows times states of the plan of the least rows times states (``cost`` itself unless a fixed cost per
    # sector chose another plan, :func:`orbital_group_plan`): what the comparison with the abelian group reads
    least_cost: float = float("nan")


def _plan_for(table, representations, members, order, characters=None, inverses=None):
    """The plan of the subgroup ``members`` (``None`` where it is not admitted).

    ``characters``: its :func:`one_dimensional_characters`, where the caller has them; ``inverses``:
    :func:`_identity_and_inverses` of ``table``.  Sums over the subgroup are formed as array products: the
    multiplicities are integers (rounded as before) and every other step works on integers, so the plan is the one
    of the loops of before.
    """

    members = sorted(members)
    if characters is None:
        characters = one_dimensional_characters(table, members)
    if not characters:
        return None
    size = len(members)
    member_array = np.asarray(members, dtype=np.int64)
    # the characters on the members (rows), and those of the representations of G on them
    signs = np.array([[chi[h] for h in members] for chi in characters], dtype=np.int64)
    restricted = np.array([rep.character[member_array] for rep in representations], dtype=np.float64)
    multiplicity = (signs.astype(np.float64) @ restricted.T) / size
    if not np.allclose(multiplicity, np.rint(multiplicity), atol=1e-6):
        raise RuntimeError("restriction multiplicities are not integers")
    multiplicity = np.rint(multiplicity).astype(np.int64)
    found = multiplicity.sum(axis=0)
    if np.any(found < 1):
        return None
    dimensions = np.array([rep.dimension for rep in representations])
    weights = []
    for row in multiplicity:
        held = np.flatnonzero(row)
        candidates = {float(dimensions[g]) / float(found[g]) for g in held}
        if len(candidates) != 1:
            return None
        weight = candidates.pop()
        if abs(weight - round(weight)) > 1e-9:
            return None
        weights.append(int(round(weight)))
    # sectors exchanged by the normalizer of H in G hold the same levels
    identity, inverse = _identity_and_inverses(table) if inverses is None else inverses
    in_members = np.zeros(order, dtype=bool)
    in_members[member_array] = True
    every = np.arange(order, dtype=np.int64)
    # g h g^-1 for every g (rows) and h of H: g normalizes H where every one lies in H (conjugation is one to one)
    conjugates = table[table[every[:, None], member_array[None, :]], inverse[:, None]]
    normalizer = np.flatnonzero(np.all(in_members[conjugates], axis=1))
    position = np.full(order, -1, dtype=np.int64)
    position[member_array] = np.arange(size)
    keys = signs.astype(np.int8)
    index_of = {keys[i].tobytes(): i for i in range(len(characters))}
    # the image of a character under g: h -> chi(g^-1 h g), on the members
    images = position[table[table[inverse[normalizer][:, None], member_array[None, :]], normalizer[:, None]]]
    orbit_of = [-1] * len(characters)
    orbits = []
    for i in range(len(characters)):
        if orbit_of[i] >= 0:
            continue
        orbit = {index_of[row.tobytes()] for row in keys[i][images]}
        for j in orbit:
            orbit_of[j] = len(orbits)
        orbits.append(sorted(orbit))
    solved = [orbit[0] for orbit in orbits]
    # A representation of G held by several solved states (summed over the solved sectors: a merged orbit holds
    # its partners in one state) has a level whose partners take one occupation each.
    solved_counts = multiplicity[solved].sum(axis=0)
    partner_irreps = tuple(
        tuple(int(g) for g in np.flatnonzero((multiplicity[i] > 0) & (solved_counts > 1))) for i in solved
    )
    levels = np.array([rep.levels for rep in representations])
    states = multiplicity @ levels / order
    cost = float(sum(states[i] for i in solved)) / size
    # right cosets G = union H c
    cosets = [identity]
    covered = np.zeros(order, dtype=bool)
    covered[table[member_array, identity]] = True
    for g in range(order):
        if not covered[g]:
            cosets.append(g)
            covered[table[member_array, g]] = True
    return OrbitalGroupPlan(
        subgroup=tuple(members),
        characters=tuple(characters[i] for i in solved),
        weights=tuple(weights[i] * len(orbits[k]) for k, i in enumerate(solved)),
        cosets=tuple(cosets),
        cost=cost,
        abelian_cost=float("nan"),
        partner_irreps=partner_irreps,
    )


def _is_reflection_group(table: np.ndarray, members) -> bool:
    identity, _ = _identity_and_inverses(table)
    return all(int(table[m, m]) == identity for m in members)


IRREP_UNIT_MODES = ("once", "split")


def subgroup_sector_cosets(table: np.ndarray, members, cosets, projection, character=None) -> tuple[int, ...] | None:
    """The cosets of a larger subgroup ``K`` whose one-dimensional sector the projected unit is, or ``None``.

    ``projection`` holds the coefficients ``b_c`` of the unit's projector ``sum_c b_c T_c`` on its sector of ``H``
    (``members``), one per right coset representative ``cosets[c]`` (the identity first).  The projector onto the
    sector ``mu`` of a subgroup ``K = union_{c in S} H c`` (``mu`` restricted to ``H`` the sector's character
    ``lambda``) is ``(1/|K|) sum_{k in K} mu(k) T_k``, whose coefficients are ``b_c = mu(c) |H| / |K|`` on ``S`` and
    zero elsewhere; two orthogonal projectors onto the same range are equal, so a unit whose coefficients have that
    form is that sector of ``K`` (Td in D2d: A1 and A2 are the sectors A1 and A2 of Td itself, ``b = (1, +-1, 1)/3``;
    E, ``(2, 1, -1)/3``, is no such sector).  The positions ``S`` (into ``cosets``, the identity first) where the
    nonzero coefficients are all ``+-1/|S|``, the identity's positive, the elements ``h c`` (``c`` in ``S``) close
    into a group and ``mu(h c) = lambda(h) sign(b_c)`` is a character of it (``character``: ``lambda`` on ``H``, an
    element index -> +-1 dict; not checked without it); ``None`` otherwise.
    """

    values = np.asarray(projection, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(values) > 1.0e-12)
    count = int(nonzero.size)
    if count < 2 or nonzero[0] != 0 or values[0] <= 0.0:
        return None
    if not np.allclose(np.abs(values[nonzero]), 1.0 / count, rtol=0.0, atol=1.0e-12):
        return None
    members = [int(h) for h in members]
    sign = {}
    elements = set()
    for position in nonzero:
        coset = int(cosets[int(position)])
        for h in members:
            element = int(table[h, coset])
            if element in elements:
                return None
            elements.add(element)
            if character is not None:
                sign[element] = int(character[h]) * (1 if values[position] > 0.0 else -1)
    if len(elements) != count * len(members):
        return None
    group = sorted(elements)
    for a in group:
        for b in group:
            product = int(table[a, b])
            if product not in elements:
                return None
            if character is not None and sign[product] != sign[a] * sign[b]:
                return None
    return tuple(int(position) for position in nonzero)


def irrep_units(table: np.ndarray, plan: OrbitalGroupPlan, mode: str = "once",
                representations: list[RealRepresentation] | None = None) -> tuple[IrrepUnit, ...] | None:
    """Every irreducible representation of ``G`` solved in one sector of ``plan``: the units of the sectors.

    A solved sector ``lambda`` of the weighted plan holds every representation ``Gamma`` of ``G`` whose restriction
    to ``H`` contains ``lambda`` (``m(lambda, Gamma) >= 1``), and a level of ``Gamma`` is found in each of the
    ``n1(Gamma)`` sectors that hold it: the E levels of Td in both A1 and B1 of D2d.  Here every ``Gamma`` is
    assigned to one solved sector (the least ``m``; then a sector other than the totally symmetric one; then the
    sector that holds the fewest representations; then the first), and its states there count ``d / m`` times.
    A sector keeps the representations assigned to it and loses the others to a projector (:class:`IrrepUnit`):

    * ``once``: one unit per weight of the representations a sector keeps; a sector that keeps all it holds at
      one weight is the sector of before (no projector);
    * ``split``: one unit per representation (the Rayleigh-Ritz work of a sector that holds two representations at
      one weight falls by half; the 3-dimensional representations of the 192 operations of a diamond supercell).

    ``None`` where a weight ``d / m`` is not an integer.  Td in D2d: A1 in A1 (E projected out), A2 and E in B1
    (one unit each), T1 in A2, T2 in B2: rows ``N/8`` each, states ``M/24, M/24, M/12, M/8, M/8`` with weights
    1, 1, 2, 3, 3 instead of four sectors of ``M/8`` (rows times states 1.2 and rows times states squared 1.5
    times less).
    """

    if mode not in IRREP_UNIT_MODES:
        raise ValueError(f"the irreducible-representation units are {' or '.join(IRREP_UNIT_MODES)}, not {mode!r}")
    if representations is None:
        representations = real_irreducible_characters(table)
    order = table.shape[0]
    members = list(plan.subgroup)
    size = len(members)
    identity, _ = _identity_and_inverses(table)
    multiplicity = np.array(
        [[sum(chi[h] * rep.character[h] for h in members) / size for rep in representations] for chi in plan.characters]
    )
    if not np.allclose(multiplicity, np.rint(multiplicity), atol=1e-6):
        raise RuntimeError("restriction multiplicities are not integers")
    multiplicity = np.rint(multiplicity).astype(np.int64)
    trivial = [all(int(chi[h]) == 1 for h in members) for chi in plan.characters]
    held = [tuple(int(g) for g in np.flatnonzero(row)) for row in multiplicity]
    assigned: list[list[int]] = [[] for _ in plan.characters]
    for g, rep in enumerate(representations):
        candidates = [s for s in range(len(plan.characters)) if multiplicity[s, g] >= 1]
        if not candidates:
            raise RuntimeError("a representation of the group is held by no solved sector")
        best = min(candidates, key=lambda s: (int(multiplicity[s, g]), trivial[s], len(held[s]), s))
        weight = rep.dimension / float(multiplicity[best, g])
        if abs(weight - round(weight)) > 1e-9:
            return None
        assigned[best].append(g)

    def coefficients(sector: int, kept: tuple[int, ...]) -> tuple[float, ...]:
        chi = plan.characters[sector]
        values = []
        for coset in plan.cosets:
            total = 0.0
            for g in kept:
                rep = representations[g]
                total += rep.levels / order * sum(rep.character[int(table[h, coset])] * chi[h] for h in members)
            values.append(float(total))
        return tuple(values)

    units: list[IrrepUnit] = []
    for sector, kept in enumerate(assigned):
        if not kept:
            continue
        weights = {g: int(round(representations[g].dimension / multiplicity[sector, g])) for g in kept}
        if mode == "split":
            groups = [(g,) for g in kept]
        else:
            groups = []
            for weight in sorted(set(weights.values())):
                groups.append(tuple(g for g in kept if weights[g] == weight))
        for group in groups:
            whole = tuple(sorted(group)) == held[sector]
            share = float(sum(multiplicity[sector, g] * representations[g].levels for g in group)) / order
            projection = None if whole else coefficients(sector, tuple(group))
            units.append(IrrepUnit(
                sector=sector,
                representations=tuple(group),
                weight=weights[group[0]],
                share=share,
                projection=projection,
                partners=tuple(int(g) for g in group if int(multiplicity[sector, g]) > 1),
                subgroup_cosets=None if projection is None else subgroup_sector_cosets(
                    table, members, plan.cosets, projection, plan.characters[sector]),
            ))
    # the identity coset first (orbital_group_plan): its coefficient is the share of the sector's states the unit keeps
    assert plan.cosets[0] == identity
    return tuple(units)


def irrep_unit_costs(plan: OrbitalGroupPlan, partner_rows: bool = False) -> tuple[float, float]:
    """Rows times states and rows times states squared of ``plan.units`` (units of ``N M`` and ``N M^2``): the
    filter-like and the Rayleigh-Ritz-like work, every unit on the rows of its sector (``N / |H|``), or with
    ``partner_rows`` a unit that is a sector of a larger subgroup ``K`` (:attr:`IrrepUnit.subgroup_cosets`) on the
    orbits of ``K`` (``N / |K|``, ``PARSEC_ORBITAL_IRREP_ROWS=partner``)."""

    if plan.units is None:
        raise ValueError("the plan holds no units")
    size = len(plan.subgroup)

    def rows(unit: IrrepUnit) -> float:
        cosets = unit.subgroup_cosets if partner_rows else None
        return 1.0 / (size * (len(cosets) if cosets else 1))

    return (float(sum(rows(unit) * unit.share for unit in plan.units)),
            float(sum(rows(unit) * unit.share ** 2 for unit in plan.units)))


def abelian_split_sectors(
    table: np.ndarray,
    plan: OrbitalGroupPlan,
    abelian_members,
    representations: list[RealRepresentation] | None = None,
    *,
    units: bool = False,
) -> tuple[bool, ...]:
    """Per solved sector of ``plan`` (per unit with ``units``), whether every representation of ``G`` it holds has
    partners in two or more sectors of the abelian subgroup ``abelian_members`` (element indices of ``table``).

    The restriction of ``Gamma`` to an abelian group of involutions ``A`` is a sum of its real characters ``chi``,
    ``n_chi = (1 / |A|) sum_a chi(a) chi_Gamma(a)``; its partners lie in the sectors ``chi`` with ``n_chi > 0``.
    A sector of the plan holds the representations with ``m(lambda, Gamma) > 0`` (a unit those it names).  A
    one-dimensional ``Gamma`` has one partner, so a sector that holds one is not split; the t2 of Td splits over
    B1, B2, B3 of D2, its e lies in A alone (the sectors A1 and B1 of D2d hold e and are not split, B2 holds t2 alone
    and is).  The spin guard of the weighted sectors asks it (``SCF/spin_sectors.abelian_partner_labels``).
    """

    if representations is None:
        representations = real_irreducible_characters(table)
    members = sorted(int(m) for m in abelian_members)
    abelian_characters = one_dimensional_characters(table, members)
    size = len(members)
    spread = []
    for rep in representations:
        counts = [sum(chi[a] * rep.character[a] for a in members) / size for chi in abelian_characters]
        spread.append(sum(1 for value in counts if value > 0.5))
    if units:
        if plan.units is None:
            raise ValueError("the plan has no units")
        return tuple(all(spread[g] >= 2 for g in unit.representations) for unit in plan.units)
    subgroup = list(plan.subgroup)
    split = []
    for chi in plan.characters:
        held = [
            g
            for g, rep in enumerate(representations)
            if sum(chi[h] * rep.character[h] for h in subgroup) / len(subgroup) > 0.5
        ]
        split.append(bool(held) and all(spread[g] >= 2 for g in held))
    return tuple(split)


def orbital_group_plan(
    operations, modulus=None, required=(), units: str | None = None, sector_cost: float = 0.0
) -> OrbitalGroupPlan:
    """The admitted subgroup of ``operations`` (a point group of signed permutations) of the least cost.

    Ties go to the larger subgroup and then to the earlier one in the enumeration order.  The abelian route
    of before (the largest subgroup whose elements are their own inverses, all of its characters, weight 1)
    is always admitted, so a plan always exists; its cost is reported as ``abelian_cost``.  ``modulus``: the
    operations are affine, with translations modulo it (:func:`multiplication_table`).  ``required``: indices of
    operations every candidate subgroup contains (the pure translations of a cell that repeats a smaller one:
    a normal subgroup of real characters, in the chosen subgroup of every cell measured; the search over the
    subgroups that contain it takes a tenth of a second instead of seconds for diamond's 192 operations).
    ``units`` (``once`` or ``split``): the plan also carries the units of :func:`irrep_units` for its sectors
    (``None`` there where they do not exist); the chosen subgroup and sectors are the same.

    ``sector_cost``: a fixed cost of every solved sector in units of ``N M`` (``PARSEC_SECTOR_OVERHEAD``,
    :func:`.axis_reflection.sector_overhead_entries`): the plan of the least ``cost + sector_cost x sectors`` is
    chosen, among the subgroups of more than one element (the trivial one is the full grid, which is not a plan of
    sectors).  ``0`` (the default) chooses as before.  ``cost`` stays the rows times states of the chosen plan.
    """

    table = multiplication_table(operations, modulus=modulus)
    order = table.shape[0]
    representations = real_irreducible_characters(table)
    inverses = _identity_and_inverses(table)
    sector_cost = float(sector_cost)
    best = None
    best_total = None
    least = None
    abelian = None
    for members in subgroups(table, required):
        if sector_cost > 0.0 and len(members) <= 1:
            continue
        characters = one_dimensional_characters(table, members)
        plan = _plan_for(table, representations, members, order, characters=characters, inverses=inverses)
        if plan is None:
            continue
        if _is_reflection_group(table, members) and (abelian is None or len(members) > len(abelian.subgroup)):
            # the former route: every sector solved (no merging) at weight 1
            abelian = OrbitalGroupPlan(tuple(sorted(members)), tuple(characters), (1,) * len(characters),
                                       plan.cosets, 1.0 / len(members), float("nan"))
        if (
            least is None
            or plan.cost < least.cost - 1e-12
            or (abs(plan.cost - least.cost) <= 1e-12 and len(plan.subgroup) > len(least.subgroup))
        ):
            least = plan
        # (with sector_cost 0 the total is the cost itself: the plan of the least cost, the choice of before)
        total = plan.cost + sector_cost * len(plan.characters) if sector_cost > 0.0 else plan.cost
        if (
            best is None
            or total < best_total - 1e-12
            or (abs(total - best_total) <= 1e-12 and len(plan.subgroup) > len(best.subgroup))
        ):
            best, best_total = plan, total
    if best is None:
        raise RuntimeError("no orbital subgroup was admitted")
    plan = OrbitalGroupPlan(best.subgroup, best.characters, best.weights, best.cosets, best.cost,
                            abelian.cost if abelian is not None else float("nan"), best.partner_irreps,
                            least_cost=least.cost)
    if units is not None:
        plan = replace(plan, units=irrep_units(table, plan, units, representations))
    return plan


__all__ = [
    "IRREP_UNIT_MODES",
    "IrrepUnit",
    "OrbitalGroupPlan",
    "RealRepresentation",
    "abelian_split_sectors",
    "conjugacy_classes",
    "irrep_unit_costs",
    "irrep_units",
    "multiplication_table",
    "one_dimensional_characters",
    "orbital_group_plan",
    "real_irreducible_characters",
    "subgroup_sector_cosets",
    "subgroups",
]
