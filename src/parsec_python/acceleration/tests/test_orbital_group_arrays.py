"""The group theory of the orbital plan as array operations (``Symmetry/orbital_groups.py``): the plan of before.

The multiplication table, the conjugacy classes, the class coefficients of the characters, the one-dimensional
characters of a subgroup and the plan of a subgroup were loops over pairs of elements in Python: 3 to 5 seconds of
the preparation of a cell whose group has 384 operations (rock-salt MgO, NaCl, fcc Al and perovskite 2x2x2 cells),
the largest stage before their SCF.  They are array operations now.  These tests keep the loops of before as the
reference and compare every result exactly (tables, classes, characters with the order of their elements, the plan,
its units), on point groups of clusters and on the groups of periodic cells with their translations (CPU).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Symmetry import orbital_groups as og
from parsec_python.acceleration.tests.test_orbital_groups import GROUPS, _FCC, _periodic_model


# ----------------------------------------------------------------------------- the loops of before (reference)
def _reference_table(operations, modulus=None):
    matrices = [og._reduced(np.asarray(np.rint(op), dtype=np.int64), modulus) for op in operations]
    index = {og._key(m): i for i, m in enumerate(matrices)}
    if len(index) != len(matrices):
        raise ValueError("the operations are not distinct")
    order = len(matrices)
    table = np.empty((order, order), dtype=np.int64)
    for i, a in enumerate(matrices):
        for j, b in enumerate(matrices):
            k = index.get(og._key(og._reduced(a @ b, modulus)))
            if k is None:
                raise ValueError("the operations are not closed under multiplication")
            table[i, j] = k
    return table


def _reference_classes(table):
    identity, inverse = og._identity_and_inverses(table)
    order = table.shape[0]
    seen = np.zeros(order, dtype=bool)
    classes = []
    for g in [identity] + [i for i in range(order) if i != identity]:
        if seen[g]:
            continue
        members = sorted({int(table[table[h, g], inverse[h]]) for h in range(order)})
        seen[members] = True
        classes.append(tuple(members))
    return classes


def _reference_coefficients(table, classes):
    order = table.shape[0]
    count = len(classes)
    class_of = np.empty(order, dtype=np.int64)
    for c, members in enumerate(classes):
        class_of[list(members)] = c
    sizes = np.array([len(members) for members in classes], dtype=np.float64)
    coefficients = np.zeros((count, count, count))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            tally = np.zeros(count)
            for x in ci:
                for y in cj:
                    tally[class_of[table[x, y]]] += 1
            coefficients[i, j] = tally / sizes
    return coefficients


def _reference_characters(table, members):
    members = sorted(int(m) for m in members)
    identity, _ = og._identity_and_inverses(table)
    generators = []
    span = og._closure(table, ())
    for g in members:
        if g not in span:
            generators.append(g)
            span = og._closure(table, generators)
    characters = []
    for signs in np.ndindex(*([2] * len(generators))):
        value = {identity: 1}
        frontier = [identity]
        consistent = True
        while frontier and consistent:
            nxt = []
            for x in frontier:
                for g, bit in zip(generators, signs):
                    y = int(table[x, g])
                    v = value[x] * (-1 if bit else 1)
                    if y in value:
                        if value[y] != v:
                            consistent = False
                            break
                    else:
                        value[y] = v
                        nxt.append(y)
                if not consistent:
                    break
            frontier = nxt
        if not consistent or len(value) != len(members):
            continue
        if all(value[int(table[a, b])] == value[a] * value[b] for a in members for b in members):
            characters.append(value)
    return characters


def _reference_plan_for(table, representations, members, order):
    members = sorted(members)
    characters = _reference_characters(table, members)
    if not characters:
        return None
    size = len(members)
    multiplicity = np.array(
        [[sum(chi[h] * rep.character[h] for h in members) / size for rep in representations] for chi in characters]
    )
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
    identity, inverse = og._identity_and_inverses(table)
    member_set = set(members)
    normalizer = [g for g in range(order) if {int(table[table[g, h], inverse[g]]) for h in members} == member_set]
    keys = [tuple(chi[h] for h in members) for chi in characters]
    index_of = {key: i for i, key in enumerate(keys)}
    orbit_of = [-1] * len(characters)
    orbits = []
    for i, chi in enumerate(characters):
        if orbit_of[i] >= 0:
            continue
        orbit = set()
        for g in normalizer:
            image = tuple(chi[int(table[table[inverse[g], h], g])] for h in members)
            orbit.add(index_of[image])
        for j in orbit:
            orbit_of[j] = len(orbits)
        orbits.append(sorted(orbit))
    solved = [orbit[0] for orbit in orbits]
    solved_counts = multiplicity[solved].sum(axis=0)
    partner_irreps = tuple(
        tuple(int(g) for g in np.flatnonzero((multiplicity[i] > 0) & (solved_counts > 1))) for i in solved
    )
    levels = np.array([rep.levels for rep in representations])
    states = multiplicity @ levels / order
    cost = float(sum(states[i] for i in solved)) / size
    cosets = [identity]
    covered = set(int(table[h, identity]) for h in members)
    for g in range(order):
        if g not in covered:
            cosets.append(g)
            covered |= {int(table[h, g]) for h in members}
    return (tuple(members), tuple(list(characters[i].items()) for i in solved),
            tuple(weights[i] * len(orbits[k]) for k, i in enumerate(solved)), tuple(cosets), cost, partner_irreps)


# ----------------------------------------------------------------------------- the groups of the comparison
def _captured_plan_inputs(grid, atoms, environment=None):
    """The operations, modulus and required members that detect_point_group_reduction hands the plan."""

    from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction

    seen = {}
    original = og.orbital_group_plan

    def spy(operations, modulus=None, required=(), units=None):
        seen.update(operations=[np.asarray(np.rint(o), dtype=np.int64) for o in operations], modulus=modulus,
                    required=tuple(required))
        return original(operations, modulus=modulus, required=required, units=units)

    with mock.patch.dict(os.environ, environment or {}), mock.patch.object(og, "orbital_group_plan", spy):
        detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
    return seen


def _rocksalt(repeat, side=6.0, points=12):
    from parsec_python.models import Atom

    atoms = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                cell = np.array((i, j, k), dtype=float)
                atoms += [Atom("Na", tuple(side * (cell + np.array(b)))) for b in _FCC]
                atoms += [Atom("Cl", tuple(side * (cell + np.array(b) + np.array((0.5, 0.0, 0.0)))))
                          for b in _FCC]
    grid, _ = _periodic_model((side * repeat,) * 3, points * repeat, 0.5, atoms, {"Na": 1.0, "Cl": 2.5})
    return grid, atoms


def _diamond(repeat, side=6.0, points=12):
    from parsec_python.models import Atom

    atoms = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                cell = np.array((i, j, k), dtype=float)
                atoms += [Atom("C", tuple(side * (cell + np.array(b) + s))) for s in (0.0, 0.25) for b in _FCC]
    grid, _ = _periodic_model((side * repeat,) * 3, points * repeat, 0.0, atoms, {"C": 2.0})
    return grid, atoms


def _cases():
    for name in ("Td", "Oh", "D4h", "Th"):
        yield name, {"operations": GROUPS[name], "modulus": None, "required": ()}
    yield "diamond cell", _captured_plan_inputs(*_diamond(1))
    yield "rock-salt cell", _captured_plan_inputs(*_rocksalt(1))
    yield "diamond 2x2x2", _captured_plan_inputs(*_diamond(2))
    yield "rock-salt 2x2x2", _captured_plan_inputs(*_rocksalt(2))


class OrbitalGroupArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = list(_cases())

    def test_the_groups_of_the_comparison(self) -> None:
        sizes = {name: len(case["operations"]) for name, case in self.cases}
        self.assertEqual(sizes["Oh"], 48)
        # the cells take their translations: affine operations with a modulus
        self.assertIsNotNone(dict(self.cases)["rock-salt 2x2x2"]["modulus"])
        self.assertGreaterEqual(sizes["rock-salt 2x2x2"], 384)

    def test_tables_classes_and_class_coefficients(self) -> None:
        for name, case in self.cases:
            with self.subTest(group=name):
                table = og.multiplication_table(case["operations"], modulus=case["modulus"])
                np.testing.assert_array_equal(table, _reference_table(case["operations"], case["modulus"]))
                classes = og.conjugacy_classes(table)
                self.assertEqual(classes, _reference_classes(table))
                # the representations of the class coefficients of before, bit for bit
                expected = _reference_coefficients(table, classes)
                with mock.patch.object(np, "tensordot", wraps=np.tensordot) as spy:
                    representations = og.real_irreducible_characters(table)
                np.testing.assert_array_equal(spy.call_args.args[1], expected)
                self.assertTrue(representations)

    def test_characters_of_every_subgroup_in_the_order_of_before(self) -> None:
        for name, case in self.cases:
            table = og.multiplication_table(case["operations"], modulus=case["modulus"])
            groups = og.subgroups(table, case["required"])
            # every subgroup of the small groups, a spread of those of the large ones
            step = max(1, len(groups) // 40)
            for members in groups[::step]:
                with self.subTest(group=name, size=len(members)):
                    found = og.one_dimensional_characters(table, members)
                    expected = _reference_characters(table, members)
                    self.assertEqual([list(c.items()) for c in found], [list(c.items()) for c in expected])

    def test_the_plan_of_every_subgroup_and_the_chosen_plan(self) -> None:
        for name, case in self.cases:
            table = og.multiplication_table(case["operations"], modulus=case["modulus"])
            order = table.shape[0]
            representations = og.real_irreducible_characters(table)
            groups = og.subgroups(table, case["required"])
            step = max(1, len(groups) // 40)
            for members in groups[::step]:
                with self.subTest(group=name, size=len(members)):
                    plan = og._plan_for(table, representations, members, order)
                    expected = _reference_plan_for(table, representations, members, order)
                    if expected is None:
                        self.assertIsNone(plan)
                        continue
                    self.assertEqual(
                        (plan.subgroup, tuple(list(c.items()) for c in plan.characters), plan.weights, plan.cosets,
                         plan.cost, plan.partner_irreps),
                        expected,
                    )
            with self.subTest(group=name, plan="chosen"):
                plan = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"])
                best = None
                for members in groups:
                    candidate = _reference_plan_for(table, representations, members, order)
                    if candidate is None:
                        continue
                    if (
                        best is None
                        or candidate[4] < best[4] - 1e-12
                        or (abs(candidate[4] - best[4]) <= 1e-12 and len(candidate[0]) > len(best[0]))
                    ):
                        best = candidate
                self.assertEqual(
                    (plan.subgroup, tuple(list(c.items()) for c in plan.characters), plan.weights, plan.cosets,
                     plan.cost, plan.partner_irreps),
                    best,
                )
                for units in ("once", "split"):
                    with_units = og.orbital_group_plan(
                        case["operations"], modulus=case["modulus"], required=case["required"], units=units
                    )
                    self.assertEqual(with_units.subgroup, plan.subgroup)
                    self.assertEqual(
                        repr(with_units.units), repr(og.irrep_units(table, plan, units, representations))
                    )

    def test_refusals_of_before(self) -> None:
        operations = list(GROUPS["D4h"])
        with self.assertRaisesRegex(ValueError, "not distinct"):
            og.multiplication_table(operations + [operations[3]])
        with self.assertRaisesRegex(ValueError, "not closed"):
            og.multiplication_table([operations[0], GROUPS["Oh"][5], GROUPS["Oh"][17]])
        # a set of members that is no subgroup has no characters, as before
        table = og.multiplication_table(GROUPS["Oh"])
        subgroup = og.subgroups(table)[5]
        outside = next(g for g in range(table.shape[0]) if g not in subgroup)
        some = sorted(subgroup | {outside})
        self.assertNotEqual(len(og._closure(table, some)), len(some))
        self.assertEqual(og.one_dimensional_characters(table, some), [])
        self.assertEqual(_reference_characters(table, some), [])


if __name__ == "__main__":
    unittest.main()
