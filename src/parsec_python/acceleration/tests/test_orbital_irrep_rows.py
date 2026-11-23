"""Units of irreducible representations on the orbits of a larger subgroup (PARSEC_ORBITAL_IRREP_ROWS, CPU).

A projected unit whose projector is ``(1/[K:H]) sum_c mu(c) T_c`` over the cosets of a subgroup ``K`` is the sector
``mu`` of ``K``: ``P = E E^T`` with ``E`` the isometry from the functions of that sector to the coefficients of the
sector of ``H``.  The partner rows reproduce the unit's spectrum (each level of the unit once) with ``[K:H]`` times
fewer rows, their expansion is that of the sector of ``H``, and the units that are no such sector (the E of Td) keep
their rows.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.tests.test_orbital_groups import _FCC, GROUPS, _periodic_model


def _unit_levels(decomposition, hamiltonian, index, count):
    """The lowest ``count`` levels of unit ``index`` on the rows of its sector of ``H`` (projected)."""

    dense = decomposition.reduce_operators(hamiltonian, (index,))[index].toarray()
    projector = decomposition.project_vectors(np.eye(dense.shape[0]), index)
    values, vectors = np.linalg.eigh(projector)
    basis = vectors[:, values > 0.5]
    return np.linalg.eigvalsh(basis.T @ dense @ basis)[:count]


def _check_partner_units(case, decomposition, hamiltonian, units, count=12):
    """Every partner unit against its projected sector, the others unchanged; ``units`` partner units, each with
    more than 1.4 times fewer rows than its sector of ``H``."""

    partner_units = decomposition.partner_units
    case.assertEqual(len(partner_units), units)
    for index in partner_units:
        case.assertGreater(decomposition.sector_size(index) / decomposition.solved_size(index), 1.4)
    for index in range(decomposition.representation_count):
        if index not in partner_units:
            case.assertEqual(decomposition.solved_size(index), decomposition.sector_size(index))
            case.assertIsNone(decomposition.partner_rows(index))
            case.assertIsNone(decomposition.partner_decomposition(index))
            continue
        rows = decomposition.partner_rows(index)
        case.assertEqual(rows.rows, decomposition.sector_rank(index))
        size = decomposition.sector_size(index)
        # E^T E = I and E E^T = P
        identity = np.eye(rows.rows)
        expanded = rows.expand(identity)
        np.testing.assert_allclose(expanded.T @ expanded, identity, rtol=0.0, atol=1e-13)
        projector = decomposition.project_vectors(np.eye(size), index)
        np.testing.assert_allclose(expanded @ expanded.T, projector, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(rows.restrict(expanded), identity, rtol=0.0, atol=1e-13)
        # the sector of K: the same full-grid functions as the sector of H of E x
        partner = decomposition.partner_decomposition(index)
        case.assertEqual(partner.sector_size(0), rows.rows)
        vector = np.random.default_rng(index).standard_normal(rows.rows)
        np.testing.assert_allclose(
            partner.expand_vector(vector, 0), decomposition.expand_vector(rows.expand(vector), index),
            rtol=0.0, atol=1e-13,
        )
        orbits, sources, values = decomposition.partner_expansion(index)
        wedge = np.zeros(decomposition.wedge_size)
        wedge[orbits] = values * vector[sources]
        np.testing.assert_allclose(
            wedge[decomposition.reduction.full_to_wedge] * decomposition.phases[index],
            partner.expand_vector(vector, 0), rtol=0.0, atol=1e-13,
        )
        # the density of a partner vector summed on the representatives' orbits of H (one per row of K) and
        # averaged over the full group is that of the sector vector
        full = partner.expand_vector(vector, 0)
        exact = decomposition.reduction.symmetrize_values(
            np.bincount(decomposition.reduction.full_to_wedge, weights=full * full, minlength=decomposition.wedge_size)
            / decomposition.reduction.multiplicities
        )
        concentrated = np.zeros(decomposition.wedge_size)
        first_orbits = decomposition.sector_orbit_indices(index)[rows.first]
        concentrated[first_orbits] = vector * vector / decomposition.reduction.multiplicities[first_orbits]
        np.testing.assert_allclose(decomposition.reduction.symmetrize_values(concentrated), exact,
                                   rtol=1e-12, atol=1e-15)
        # the reduced operator of the sector of K has the unit's levels; so do its stencil and its projectors
        reduced = partner.reduce_operators(hamiltonian)[0].toarray()
        np.testing.assert_allclose(reduced, expanded.T @ decomposition.reduce_operators(
            hamiltonian, (index,))[index].toarray() @ expanded, rtol=0.0, atol=1e-11)
        levels = min(count, rows.rows)
        np.testing.assert_allclose(np.linalg.eigvalsh(reduced)[:levels],
                                   _unit_levels(decomposition, hamiltonian, index, levels), rtol=0.0, atol=1e-10)


class PartnerCosetTests(unittest.TestCase):
    def test_the_units_that_are_sectors_of_a_larger_subgroup(self):
        from parsec_python.acceleration.Symmetry.axis_reflection import _affine_matrix
        from parsec_python.acceleration.Symmetry.orbital_groups import irrep_unit_costs, orbital_group_plan

        # Td: A1 and A2 are the sectors of Td itself (3 cosets of D2d), E is none
        plan = orbital_group_plan(GROUPS["Td"], units="once")
        found = sorted((unit.weight, None if unit.subgroup_cosets is None else len(unit.subgroup_cosets))
                       for unit in plan.units if unit.projection is not None)
        self.assertEqual(found, [(1, 3), (1, 3), (2, None)])
        self.assertTrue(all(unit.subgroup_cosets is None for unit in plan.units if unit.projection is None))
        before, ritz_before = irrep_unit_costs(plan)
        after, ritz_after = irrep_unit_costs(plan, partner_rows=True)
        # rows x states: 30 -> 26 / 576, rows x states^2 24 -> 22.67 / 4608
        self.assertAlmostEqual(before / after, 30.0 / 26.0)
        self.assertAlmostEqual(ritz_before / ritz_after, 24.0 / (22.0 + 2.0 / 3.0))
        # Td x Z2^3 (the 192 operations of an even diamond supercell), split: 12 units of weight 3 on subgroups of
        # 64, 4 of weight 1 on all 192 operations, the two E-like units on the sectors of 32
        counts = np.array([8, 8, 8])
        elements = [_affine_matrix(np.asarray(R), np.asarray(t)) for R in GROUPS["Td"]
                    for t in itertools.product((0, 4), repeat=3)]
        required = tuple(i for i, e in enumerate(elements) if np.array_equal(e[:3, :3], np.eye(3)))
        plan = orbital_group_plan(elements, modulus=counts, required=required, units="split")
        ratios = sorted(len(unit.subgroup_cosets) for unit in plan.units if unit.subgroup_cosets is not None)
        self.assertEqual(ratios, [2] * 12 + [6] * 4)
        before, ritz_before = irrep_unit_costs(plan)
        after, ritz_after = irrep_unit_costs(plan, partner_rows=True)
        self.assertGreater(before / after, 1.55)
        self.assertGreater(ritz_before / ritz_after, 1.36)

    def test_the_switch_is_validated(self):
        from parsec_python.acceleration.Symmetry.axis_reflection import orbital_irrep_rows_setting

        # partner by default since 2026-11-22 (whole first runs on A100 faster than on sector rows in every case
        # measured); sector restores the rows of before
        for value, expected in (("", "partner"), ("partner", "partner"), ("SECTOR", "sector")):
            with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREP_ROWS": value}):
                self.assertEqual(orbital_irrep_rows_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREP_ROWS": "on"}):
            with self.assertRaises(ValueError):
                orbital_irrep_rows_setting()


class TdPartnerRowTests(unittest.TestCase):
    """CH4 (Td) with Gaussian wells on half-shifted and zero-shift grids."""

    def test_partner_rows_of_the_td_units(self):
        import scipy.sparse as sp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Symmetry import (
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.models import Atom, GridSettings

        a = 1.2
        atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
            Atom("H", tuple(a * np.array(signs, dtype=float)))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        for shift in (0.5, 0.0):
            grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.8, expansion_order=4, shift=(shift,) * 3))
            points = np.asarray(grid.physical_coordinates(grid.integer_coordinates))
            potential = np.zeros(points.shape[0])
            for atom, depth in zip(atoms, (3.0, 1.0, 1.0, 1.0, 1.0)):
                potential -= depth * np.exp(-np.sum((points - np.asarray(atom.position)) ** 2, axis=1) / 0.8)
            hamiltonian = (0.5 * build_negative_laplacian(grid) + sp.diags(potential)).tocsr()
            abelian = SignedPermutationReduction.detect(grid, atoms)
            for mode in ("once", "split"):
                environment = {"PARSEC_ORBITAL_IRREPS": mode, "PARSEC_ORBITAL_IRREP_ROWS": "partner"}
                with self.subTest(shift=shift, mode=mode), patch.dict(os.environ, environment):
                    reduction = detect_point_group_reduction(grid, atoms, abelian)
                    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                    _check_partner_units(self, decomposition, hamiltonian, 2)
                    # the switch keeps the rows of the sectors (read once per decomposition), and so does a
                    # decomposition told to keep them
                    with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREP_ROWS": "sector"}):
                        rows = ReflectionRepresentationDecomposition.build(grid, reduction)
                        self.assertEqual(rows.partner_units, ())
                        self.assertEqual(rows.solved_sizes, rows.sector_sizes)
                    self.assertEqual(rows.partner_units, ())
                    self.assertEqual(len(decomposition.partner_units), 2)
                    kept = ReflectionRepresentationDecomposition.build(grid, reduction)
                    kept.keep_sector_rows("a test")
                    self.assertEqual(kept.partner_units, ())
                    self.assertEqual(kept.sector_rows_kept, "a test")
                # the default solves the units on partner rows
                with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": mode, "PARSEC_ORBITAL_IRREP_ROWS": ""}):
                    default = ReflectionRepresentationDecomposition.build(grid, reduction)
                    self.assertEqual(default.partner_units, decomposition.partner_units)
                    self.assertEqual(default.solved_sizes, decomposition.solved_sizes)
            with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": "off"}):
                reduction = detect_point_group_reduction(grid, atoms, abelian)
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                self.assertEqual(decomposition.partner_units, ())
                self.assertEqual(decomposition.solved_sizes, decomposition.sector_sizes)


def _check_partner_projectors(case, decomposition, label):
    """The partner rows of every unit of ``decomposition`` against its projector's tables on random vectors:
    ``E E^T X = P X``, ``E^T E Y = Y``, as many rows as the projector's trace (the tables' and the cheap one)."""

    rng = np.random.default_rng(11)
    for index in decomposition.partner_units:
        rows = decomposition.partner_rows(index)
        size = decomposition.sector_size(index)
        case.assertEqual(rows.rows, decomposition.sector_rank(index), label)
        case.assertEqual(rows.rows, decomposition.projector_trace(index), label)
        vectors = rng.standard_normal((size, 3))
        np.testing.assert_allclose(rows.expand(rows.restrict(vectors)), decomposition.project_vectors(vectors, index),
                                   rtol=0.0, atol=1e-12, err_msg=label)
        partner = rng.standard_normal((rows.rows, 3))
        np.testing.assert_allclose(rows.restrict(rows.expand(partner)), partner, rtol=0.0, atol=1e-12, err_msg=label)
        np.testing.assert_array_less(0.0, rows.values[rows.first], err_msg=label)
        # the sector of K: the same full-grid functions as the sector of H of E x
        vector = partner[:, 0]
        np.testing.assert_allclose(
            decomposition.partner_decomposition(index).expand_vector(vector, 0),
            decomposition.expand_vector(rows.expand(vector), index), rtol=0.0, atol=1e-12, err_msg=label,
        )


class GroupPartnerRowTests(unittest.TestCase):
    def test_clusters_of_the_cubic_and_tetragonal_groups(self):
        """Clusters of Oh, Th, T and D4h (two orbits of generic points each) on half-shifted and zero-shift grids:
        the partner rows of ``once`` and ``split`` (Oh: 4 units on the orbits of Oh; Th and T: units of ``split`` on
        the orbits of the whole group; D4h: 8 units of ``split`` on subgroups of index 2) against the tables of their
        projectors, with points on the mirror planes and axes."""

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import (
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.acceleration.tests.test_orbital_groups import GROUPS
        from parsec_python.models import Atom, GridSettings

        def orbit(group, point):
            seen = []
            for operation in GROUPS[group]:
                image = np.asarray(operation, dtype=float) @ np.asarray(point, dtype=float)
                if not any(np.allclose(image, other) for other in seen):
                    seen.append(image)
            return seen

        cases = {
            "Oh": ((1.7, 0.0, 0.0), (1.1, 0.9, 0.0)),
            "Th": ((1.6, 0.7, 0.0), (1.2, 1.2, 1.2)),
            "T": ((1.5, 0.6, 0.3), (1.1, 1.1, 1.1)),
            "D4h": ((1.8, 0.0, 0.0), (1.2, 0.5, 0.0)),
        }
        found = {}
        for group, points in cases.items():
            atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
                Atom(symbol, tuple(position)) for symbol, point in zip(("H", "O"), points)
                for position in orbit(group, point)
            ]
            for shift in (0.5, 0.0):
                grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=4, shift=(shift,) * 3))
                abelian = SignedPermutationReduction.detect(grid, atoms)
                for mode in ("once", "split"):
                    label = f"{group} shift {shift} {mode}"
                    with self.subTest(label), patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": mode,
                                                                      "PARSEC_ORBITAL_IRREP_ROWS": "partner"}):
                        reduction = detect_point_group_reduction(grid, atoms, abelian)
                        self.assertEqual(len(reduction.full_operations), len(GROUPS[group]), label)
                        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                        found[label] = len(decomposition.partner_units)
                        _check_partner_projectors(self, decomposition, label)
                        # the rows of the coset images that the detection keeps give the tables and partner rows of
                        # the row maps, bit for bit
                        # (kept where units have projectors only: the groups of ``once`` without any keep the
                        # reduction of before)
                        self.assertEqual(reduction.coset_rows is not None, bool(decomposition.projected_sectors),
                                         label)
                        mapped = ReflectionRepresentationDecomposition.build(
                            grid, dataclasses.replace(reduction, coset_rows=None))
                        for index in decomposition.projected_sectors:
                            for left, right in zip(decomposition.projection_tables(index),
                                                   mapped.projection_tables(index)):
                                np.testing.assert_array_equal(left, right, err_msg=label)
                            self.assertEqual(decomposition.projector_trace(index), mapped.projector_trace(index))
                        for index in decomposition.partner_units:
                            left, right = decomposition.partner_rows(index), mapped.partner_rows(index)
                            for name in ("first", "row_of_sector", "values"):
                                np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=label)
        for shift in (0.5, 0.0):
            self.assertEqual(found[f"Oh shift {shift} once"], 4)
            self.assertEqual(found[f"D4h shift {shift} split"], 8)
            self.assertEqual(found[f"D4h shift {shift} once"], 0)
            self.assertGreater(found[f"Th shift {shift} split"], 0)
            self.assertGreater(found[f"T shift {shift} split"], 0)


class PeriodicPartnerRowTests(unittest.TestCase):
    def test_diamond_and_rocksalt_cells(self):
        from parsec_python.acceleration.Symmetry import (
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.models import Atom

        side = 6.0
        rocksalt = [Atom("Na", tuple(side * np.array(b))) for b in _FCC] + [
            Atom("Cl", tuple(side * (np.array(b) + np.array((0.5, 0.0, 0.0))))) for b in _FCC
        ]
        diamond = [Atom("C", tuple(side * (np.array(b) + s))) for s in (0.0, 0.25) for b in _FCC]
        cases = (
            ("rocksalt", rocksalt, {"Na": 1.0, "Cl": 2.5}, "off", "once", 4),
            ("diamond half", diamond, {"C": 2.0}, "half", "split", 8),
            ("diamond auto", diamond, {"C": 2.0}, "auto", "split", 4),
        )
        for name, atoms, depths, translations, mode, units in cases:
            grid, hamiltonian = _periodic_model((side,) * 3, 12, 0.5, atoms, depths)
            environment = {"PARSEC_ORBITAL_IRREPS": mode, "PARSEC_PERIODIC_TRANSLATIONS": translations,
                           "PARSEC_ORBITAL_IRREP_ROWS": "partner"}
            with self.subTest(case=name), patch.dict(os.environ, environment):
                reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                _check_partner_units(self, decomposition, hamiltonian, units)

    def test_a_unit_without_a_function_on_a_coarse_grid(self):
        """The 192 operations of a rock-salt cube on 8 points per edge: one unit of ``split`` has an empty range.

        Its projector's trace is zero (counted without the tables, as every rank counts it, and equal to the trace
        of the tables for every unit), the solved sizes are counted under both row settings, and the run is refused
        before the sectors are placed, with a message that names the grid; 12 points per edge have no such unit.
        """

        from parsec_python.acceleration.Eigensolvers.symmetry import irrep_units_refusal, solved_sector_sizes
        from parsec_python.acceleration.Symmetry import (
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.models import Atom

        side = 6.0
        rocksalt = [Atom("Na", tuple(side * np.array(b))) for b in _FCC] + [
            Atom("Cl", tuple(side * (np.array(b) + np.array((0.5, 0.0, 0.0))))) for b in _FCC
        ]
        for points, translations, rows in itertools.product((8, 12), ("auto", "half"), ("sector", "partner")):
            grid, _ = _periodic_model((side,) * 3, points, 0.0, rocksalt, {"Na": 1.0, "Cl": 2.5})
            environment = {"PARSEC_ORBITAL_IRREPS": "split", "PARSEC_PERIODIC_TRANSLATIONS": translations,
                           "PARSEC_ORBITAL_IRREP_ROWS": rows}
            with self.subTest(points=points, translations=translations, rows=rows), \
                    patch.dict(os.environ, environment):
                reduction = detect_point_group_reduction(
                    grid, rocksalt, SignedPermutationReduction.detect(grid, rocksalt))
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                self.assertTrue(decomposition.projected_sectors)
                for index in decomposition.projected_sectors:
                    self.assertEqual(decomposition.projector_trace(index), decomposition.sector_rank(index))
                sizes = solved_sector_sizes(decomposition)
                refusal = irrep_units_refusal(decomposition, None)
                if points == 12:
                    self.assertEqual(decomposition.empty_units, ())
                    self.assertIsNone(refusal)
                    continue
                empty = decomposition.empty_units
                self.assertEqual(len(empty), 1)
                self.assertIn("no function on this grid", refusal)
                self.assertIn("PARSEC_ORBITAL_IRREPS=off", refusal)
                if rows == "partner":
                    # a unit on partner rows: its solved rows are the (empty) range
                    self.assertIn(empty[0], decomposition.partner_units)
                    self.assertEqual(sizes[empty[0]], 0)
                else:
                    self.assertEqual(decomposition.partner_units, ())
                    self.assertEqual(sizes[empty[0]], decomposition.sector_size(empty[0]))


if __name__ == "__main__":
    unittest.main()
