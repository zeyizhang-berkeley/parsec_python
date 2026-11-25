"""The partners of the weighted sectors of a point group in the exact exchange of a hybrid functional.

The weighted sectors (:class:`.axis_reflection.PointGroupReduction`) solve the one-dimensional sectors ``lambda`` of a
subgroup ``H`` of the point group ``G`` and count a state of sector weight ``w`` for ``w`` states.  The density of a
run is the average over ``G`` of the weighted sum of the solved states, ``rho(r) = (1/|G|) sum_g sum_s o_s
|phi_s(g^-1 r)|^2`` (``o_s = f_s w_s``, the occupation times the weight), and the density matrix that goes with it is

``rho(r, r') = sum_s (o_s / |G|) sum_g (g phi_s)(r) (g phi_s)(r') = sum_s (o_s / [G:H]) sum_c F_cs(r) F_cs(r')``,

``F_cs(r) = phi_s(c r)`` for the right coset representatives ``c`` of ``G = union H c`` (``c^-1`` runs over the left
cosets ``c^-1 H``, and ``h phi_s = lambda(h) phi_s`` makes the terms of one left coset equal).  For a level of a real
irreducible representation ``Gamma`` of real dimension ``d`` (Schur's orthogonality: ``sum_g D(g) P D(g)^T = (|G|/d)
1`` for the projector ``P`` onto one partner) this is the projector onto the level times its occupation, whatever
partner of it ``phi_s`` is, so the exchange operator of the hybrid,

``(K psi)(r) = - sum_s (o_s / [G:H]) sum_c F_cs(r) v[F_cs psi](r)``,

is exact for the weighted sectors.  A state stands for its ``[G:H]`` coset images ``F_cs`` (the identity first, the
state itself) with ``o_s / [G:H]`` each: the "images" form of a sector.  Where every representation that a solved
sector holds has all its partners among the solved states (``m(lambda, Gamma)`` summed over the solved sectors equals
``d``: the sector weight is 1), the solved partners of a level are an orthonormal basis of it and the level's term is
``o_s phi_s phi_s^T`` of those states alone: the "self" form, which needs no image (the a1, a2 and e levels of Td in the
sectors A1 and B1 of D2d; the t1 and t2 levels, one partner in A2 or B2 of weight 3, take the images under the two
threefold rotations of the cosets).

:func:`exchange_partners` gives, for a reduction, the weight and the form of every solved sector, ``[G:H]`` and the
coset representatives; the grid row map of a representative (``map[p]`` the row of ``c r_p``) gives the image of a
sector's orbitals: ``F_c[p] = phi[map[p]]`` (:class:`..Hamiltonian.exchange_device.ImageExpansion`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np


@dataclass
class ExchangePartners:
    """The partners of the solved sectors of a :class:`.axis_reflection.PointGroupReduction` in the exact exchange.

    ``weights[s]``: the weight of solved sector ``s``; ``images[s]``: whether its states stand for their coset images
    (the "images" form of the module), else for themselves ("self"); ``cosets``: ``[G:H]``; ``operations``: the right
    coset representatives (the identity first, as the reduction lists them); ``grid``: the grid of the reduction.
    """

    weights: np.ndarray
    images: np.ndarray
    cosets: int
    operations: np.ndarray
    grid: object = field(repr=False)
    _maps: dict = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def sector_count(self) -> int:
        return int(self.weights.size)

    def row_map(self, coset: int) -> np.ndarray:
        """The grid row of ``c r`` for every row ``r`` of the grid (``c`` the representative ``coset``, from 1)."""

        from .axis_reflection import _grid_row_mapping_operation

        coset = int(coset)
        if not 1 <= coset < self.cosets:
            raise ValueError("the image of a coset other than the identity's is asked for")
        with self._lock:
            found = self._maps.get(coset)
            if found is None:
                found = _grid_row_mapping_operation(self.grid, np.asarray(self.operations[coset]), 1.0e-10)
                if found is None:
                    raise RuntimeError("a coset representative of the point group no longer maps the grid")
                found = np.ascontiguousarray(found, dtype=np.int64)
                self._maps[coset] = found
        return found


def exchange_partners(reduction) -> ExchangePartners:
    """The :class:`ExchangePartners` of the weighted sectors of ``reduction`` (a cluster's point group).

    The multiplicity ``m(lambda_s, Gamma) = (1/|H|) sum_h lambda_s(h) chi_Gamma(h)`` of every real irreducible
    representation ``Gamma`` of ``G`` (:func:`.orbital_groups.real_irreducible_characters`) in every solved sector,
    checked against the weights (``sum_s m(lambda_s, Gamma) w_s = d(Gamma)``, the partners of a level counted with the
    weights); a sector whose every representation has ``sum_s m(lambda_s, Gamma) = d(Gamma)`` (all partners solved)
    takes the "self" form, any other the "images" form.  Refused: the operations of a cell with translations, and the
    units of ``PARSEC_ORBITAL_IRREPS`` (a unit holds some of the representations of its character only).
    """

    from .orbital_groups import _key, multiplication_table, real_irreducible_characters

    if getattr(reduction, "translations", None) is not None or getattr(reduction, "full_translations", None) is not None:
        raise NotImplementedError("the exchange partners of weighted sectors are formed for a cluster's point group")
    if getattr(reduction, "sector_projected", None) is not None:
        raise NotImplementedError(
            "the exchange partners of weighted sectors are formed for the sectors of a subgroup, not for the units of "
            "PARSEC_ORBITAL_IRREPS"
        )
    full = [np.asarray(np.rint(operation), dtype=np.int64) for operation in np.asarray(reduction.full_operations)]
    table = multiplication_table(full)
    representations = real_irreducible_characters(table)
    index = {_key(operation): number for number, operation in enumerate(full)}
    members = []
    for operation in np.asarray(reduction.operations):
        number = index.get(_key(np.asarray(np.rint(operation), dtype=np.int64)))
        if number is None:
            raise RuntimeError("an operation of the subgroup is not one of the point group")
        members.append(number)
    members = np.asarray(members, dtype=np.int64)
    characters = np.asarray(reduction.sector_characters, dtype=np.float64)
    if characters.ndim != 2 or characters.shape[1] != members.size:
        raise RuntimeError("the sector characters do not match the operations of the subgroup")
    restricted = np.array([rep.character[members] for rep in representations], dtype=np.float64)
    multiplicity = characters @ restricted.T / float(members.size)
    if not np.allclose(multiplicity, np.rint(multiplicity), atol=1.0e-6):
        raise RuntimeError("the restriction multiplicities of the weighted sectors are not integers")
    multiplicity = np.rint(multiplicity).astype(np.int64)
    dimensions = np.array([rep.dimension for rep in representations], dtype=np.int64)
    weights = np.asarray(reduction.sector_weights, dtype=np.float64)
    if weights.shape != (characters.shape[0],):
        raise RuntimeError("one weight per solved sector is needed")
    counted = (multiplicity * weights[:, None]).sum(axis=0)
    if not np.allclose(counted, dimensions, atol=1.0e-9):
        raise RuntimeError("the weights of the solved sectors do not count the partners of every level")
    complete = multiplicity.sum(axis=0) == dimensions
    images = np.array([not bool(np.all(complete[np.flatnonzero(row)])) for row in multiplicity], dtype=bool)
    if np.any(~images & (weights != 1.0)):
        raise RuntimeError("a sector that holds whole levels has a weight other than 1")
    operations = np.asarray(reduction.coset_operations, dtype=np.int64)
    cosets = int(operations.shape[0])
    if cosets * members.size != len(full):
        raise RuntimeError("the coset representatives do not complete the subgroup to the point group")
    if not np.array_equal(operations[0], np.eye(3, dtype=np.int64)):
        raise RuntimeError("the first coset representative is not the identity")
    if getattr(reduction, "grid", None) is None:
        raise RuntimeError("the exchange partners of weighted sectors need the grid of the reduction")
    return ExchangePartners(
        weights=np.ascontiguousarray(weights),
        images=np.ascontiguousarray(images),
        cosets=cosets,
        operations=np.ascontiguousarray(operations),
        grid=reduction.grid,
    )


__all__ = ["ExchangePartners", "exchange_partners"]
