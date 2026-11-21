"""Units of the irreducible representations of the full point group (PARSEC_ORBITAL_IRREPS, CPU).

Every representation of ``G`` solved in one sector of the weighted plan (``orbital_groups.irrep_units``), with the
projector onto it on the coefficients of the sector (``ReflectionRepresentationDecomposition.projection_tables``):
the projectors are orthogonal projectors that commute with the sector Hamiltonian, and the spectra of the units,
each level counted with its weight, are the full-grid spectrum.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.tests.test_orbital_groups import _FCC, GROUPS, _periodic_model


def _unit_spectrum(decomposition, hamiltonian, count):
    """The levels of every sector or unit, each repeated by its weight, sorted (the lowest ``count``)."""

    levels = []
    for index, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
        dense = matrix.toarray()
        if decomposition.sector_projected(index):
            projector = decomposition.project_vectors(np.eye(dense.shape[0]), index)
            values, vectors = np.linalg.eigh(projector)
            basis = vectors[:, values > 0.5]
            dense = basis.T @ dense @ basis
        values = np.linalg.eigvalsh(dense)[:count]
        levels.extend(np.repeat(values, int(decomposition.sector_weights[index])))
    return np.sort(levels)[:count]


def _check_projectors(case, decomposition, hamiltonian):
    reduced = decomposition.reduce_operators(hamiltonian)
    for index in decomposition.projected_sectors:
        size = decomposition.sector_size(index)
        projector = decomposition.project_vectors(np.eye(size), index)
        np.testing.assert_allclose(projector, projector.T, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(projector @ projector, projector, rtol=0.0, atol=1e-12)
        matrix = reduced[index].toarray()
        commutator = matrix @ projector - projector @ matrix
        case.assertLess(float(np.max(np.abs(commutator))), 1e-10 * max(1.0, float(np.max(np.abs(matrix)))))
        rank = decomposition.sector_rank(index)
        case.assertEqual(rank, int(round(float(np.trace(projector)))))
        case.assertTrue(0 < rank < size)
        # the host projection of one vector and of a block agree
        vector = np.random.default_rng(index).standard_normal(size)
        np.testing.assert_allclose(decomposition.project_vectors(vector, index), projector @ vector, atol=1e-13)


def _check_unit_placement(case, decomposition):
    """Five Td units on four GPUs of one process by cost (T1, T2, E, A1 + A2), none pooled on an MPI rank."""

    from parsec_python.acceleration.backends.implicit_stencil import FilterPlan
    from parsec_python.acceleration.Eigensolvers.symmetry import (
        sector_bases_divisible,
        sector_placement_costs,
        unit_device_ids,
    )

    plan = FilterPlan(states=240, buffer=8, applications=1)
    costs = sector_placement_costs(decomposition, plan)
    owners = unit_device_ids(decomposition, plan, (0, 1, 2, 3))
    case.assertEqual(len(owners), 5)
    loads = [sum(cost for cost, owner in zip(costs, owners) if owner == device) for device in range(4)]
    # the fullest GPU holds the costliest unit alone (in turn, GPU 0 took units 0 and 4)
    case.assertAlmostEqual(max(loads), max(costs))
    case.assertEqual(sorted(len([o for o in owners if o == device]) for device in range(4)), [1, 1, 1, 2])
    weight_one = [index for index, weight in enumerate(decomposition.sector_weights) if weight == 1]
    case.assertEqual(owners[weight_one[0]], owners[weight_one[1]])
    # one GPU, or as many GPUs as units: the devices in turn, as before
    case.assertIsNone(unit_device_ids(decomposition, plan, (0,)))
    case.assertIsNone(unit_device_ids(decomposition, plan, tuple(range(5))))
    # units that fit whole on single devices keep disjoint GPUs on an MPI rank; larger ones, or no devices named,
    # follow the rule of the sectors (here forced on), so that a rank may pool them (memory of whole units)
    from parsec_python.acceleration.Eigensolvers.symmetry import units_fit_whole

    sizes = [decomposition.sector_size(index) for index in range(5)]
    counts = [cost / size for cost, size in zip(costs, sizes)]
    need = 8.0 * (sum(costs) / 4 + 2 * max(costs))
    budget = 0.82
    case.assertTrue(all(count >= 1 for count in counts))
    case.assertTrue(units_fit_whole(decomposition, plan, (0, 1, 2, 3), int(need / budget) + 1))
    case.assertFalse(units_fit_whole(decomposition, plan, (0, 1, 2, 3), int(need / budget) - 1))
    case.assertFalse(units_fit_whole(decomposition, plan, None, 10**15))
    with patch.dict(os.environ, {"PARSEC_CUPY_DISTRIBUTED_STATE": "on"}):
        case.assertEqual(sector_bases_divisible(decomposition, plan, (0, 1, 2, 3), 10**15), (False,) * 5)
        case.assertEqual(sector_bases_divisible(decomposition, plan, (0, 1, 2, 3), 1), (True,) * 5)
        case.assertEqual(sector_bases_divisible(decomposition, plan), (True,) * 5)
    # An MPI rank counts its own units only (review of 2026-11-20: the sum of every unit of the run over the GPUs
    # of one rank pooled units that fit whole): two ranks of four GPUs, the units placed by cost.
    from types import SimpleNamespace

    from parsec_python.acceleration.Eigensolvers.symmetry import planned_owned_units
    from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext

    devices = (0, 1, 2, 3)
    case.assertIsNone(planned_owned_units(decomposition, None, devices, costs))
    case.assertIsNone(planned_owned_units(SimpleNamespace(representation_count=5), MPISectorContext(
        SimpleNamespace(rank=0, size=2)), devices, costs))
    owned_by_rank = [
        planned_owned_units(decomposition, MPISectorContext(SimpleNamespace(rank=rank, size=2)), devices, costs)
        for rank in range(2)
    ]
    case.assertEqual(sorted(index for owned in owned_by_rank for index in owned), list(range(5)))
    # a placement that fails is left to configure, which reports it on every rank
    case.assertIsNone(planned_owned_units(
        decomposition, MPISectorContext(SimpleNamespace(rank=0, size=6)), devices, costs))
    case.assertFalse(units_fit_whole(decomposition, plan, devices, 10**15, owned=()))
    for owned in owned_by_rank:
        own = [8.0 * costs[index] for index in owned]
        fits = int((sum(own) / 4 + 2 * max(own)) / budget) + 1
        case.assertTrue(units_fit_whole(decomposition, plan, devices, fits, owned=owned))
        case.assertFalse(units_fit_whole(decomposition, plan, devices, fits - 2, owned=owned))
        case.assertLess(fits, int(need / budget))
        case.assertFalse(units_fit_whole(decomposition, plan, devices, fits))
        with patch.dict(os.environ, {"PARSEC_CUPY_DISTRIBUTED_STATE": "on"}):
            case.assertEqual(sector_bases_divisible(decomposition, plan, devices, fits, owned=owned), (False,) * 5)
            case.assertEqual(sector_bases_divisible(decomposition, plan, devices, fits), (True,) * 5)


class UnitPlanTests(unittest.TestCase):
    def test_units_of_the_cubic_groups(self):
        from parsec_python.acceleration.Symmetry.orbital_groups import (
            irrep_unit_costs,
            multiplication_table,
            orbital_group_plan,
            real_irreducible_characters,
        )

        expected = {
            # group: mode -> (sorted unit weights, filter gain, Rayleigh-Ritz gain over the weighted plan)
            "Td": {"once": ([1, 1, 2, 3, 3], 1.2, 1.5), "split": ([1, 1, 2, 3, 3], 1.2, 1.5)},
            "Oh": {"once": ([1, 1, 1, 1, 2, 2, 3, 3, 3, 3], 1.2, 1.5)},
            "T": {"once": ([1, 3], 1.0, 1.0), "split": ([1, 1, 3], 1.0, 9 / 7)},
            "D4h": {"once": ([1, 1, 1, 1, 2, 2], 1.0, 1.0)},
        }
        for name, modes in expected.items():
            operations = GROUPS[name]
            table = multiplication_table(operations)
            representations = real_irreducible_characters(table)
            base = orbital_group_plan(operations)
            members = list(base.subgroup)
            rows = 1.0 / len(members)
            shares = []
            for chi in base.characters:
                multiplicity = [round(sum(chi[h] * r.character[h] for h in members) / len(members))
                                for r in representations]
                shares.append(sum(m * r.levels for m, r in zip(multiplicity, representations)) / len(operations))
            filter_before = sum(rows * share for share in shares)
            ritz_before = sum(rows * share * share for share in shares)
            for mode, (weights, filter_gain, ritz_gain) in modes.items():
                with self.subTest(group=name, mode=mode):
                    plan = orbital_group_plan(operations, units=mode)
                    self.assertEqual(plan.subgroup, base.subgroup)
                    self.assertEqual(sorted(unit.weight for unit in plan.units), weights)
                    filter_cost, ritz_cost = irrep_unit_costs(plan)
                    self.assertAlmostEqual(filter_before / filter_cost, filter_gain)
                    self.assertAlmostEqual(ritz_before / ritz_cost, ritz_gain)
                    # every representation in exactly one unit, its degeneracy counted once:
                    # sum over units of weight x multiplicity of the unit's character in Gamma|H = dimension
                    seen = sorted(g for unit in plan.units for g in unit.representations)
                    self.assertEqual(seen, list(range(len(representations))))
                    for unit in plan.units:
                        chi = plan.characters[unit.sector]
                        for g in unit.representations:
                            rep = representations[g]
                            multiplicity = round(sum(chi[h] * rep.character[h] for h in members) / len(members))
                            self.assertEqual(unit.weight * multiplicity, rep.dimension)
                    # the projector of a unit keeps its share of the sector: b_identity = share * |G| / [G:H] ...
                    for unit in plan.units:
                        if unit.projection is not None:
                            self.assertAlmostEqual(unit.projection[0], unit.share * len(members))

    def test_the_switch_is_validated(self):
        from parsec_python.acceleration.Symmetry.axis_reflection import orbital_irreps_setting

        for value, expected in (("", "off"), ("off", "off"), ("ONCE", "once"), ("split", "split")):
            with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": value}):
                self.assertEqual(orbital_irreps_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": "on"}):
            with self.assertRaises(ValueError):
                orbital_irreps_setting()


class TdUnitSpectrumTests(unittest.TestCase):
    """CH4 (Td) with Gaussian wells on half-shifted and zero-shift grids: the units against the full grid."""

    def test_units_give_the_full_spectrum(self):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Symmetry import (
            PointGroupReduction,
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
            count = 30
            full = np.sort(spla.eigsh(hamiltonian, k=count, which="SA", tol=1e-12)[0])
            before = detect_point_group_reduction(grid, atoms, abelian)
            self.assertIsNone(before.sector_projected)
            for mode in ("once", "split"):
                with self.subTest(shift=shift, mode=mode), patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": mode}):
                    reduction = detect_point_group_reduction(grid, atoms, abelian)
                    self.assertIsInstance(reduction, PointGroupReduction)
                    self.assertEqual(reduction.group_order, 8)
                    np.testing.assert_array_equal(reduction.full_to_wedge, before.full_to_wedge)
                    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                    self.assertEqual(sorted(decomposition.sector_weights.tolist()), [1, 1, 2, 3, 3])
                    self.assertEqual(len(decomposition.projected_sectors), 3)
                    # one totally symmetric unit (A1 of Td; the E of A1 of D2d is solved in B1)
                    self.assertEqual(int(np.sum(np.all(decomposition.characters == 1, axis=1))), 1)
                    np.testing.assert_allclose(sorted(decomposition.sector_shares * 24), [1, 1, 2, 3, 3])
                    # (with the partner occupations of the weighted inputs) the partner representations
                    # follow the units: the E of Td, split over A1 and B1 of D2d before (its bit in both sectors),
                    # is solved in one state per level in its unit, as every other representation here
                    self.assertTrue(any(before.sector_partner_irreps))
                    self.assertEqual(len(reduction.sector_partner_irreps), decomposition.representation_count)
                    self.assertEqual(decomposition.sector_partner_masks, (0,) * decomposition.representation_count)
                    _check_projectors(self, decomposition, hamiltonian)
                    _check_unit_placement(self, decomposition)
                    np.testing.assert_allclose(
                        _unit_spectrum(decomposition, hamiltonian, count), full, rtol=0.0, atol=1e-10
                    )
                    # the partners of the E unit's lowest state: two orthonormal eigenvectors of the full operator
                    unit = int(np.flatnonzero(decomposition.sector_weights == 2)[0])
                    matrix = decomposition.reduce_operators(hamiltonian)[unit].toarray()
                    projector = decomposition.project_vectors(np.eye(matrix.shape[0]), unit)
                    values, vectors = np.linalg.eigh(projector @ matrix @ projector + 50.0 * (np.eye(matrix.shape[0]) - projector))
                    state = decomposition.expand_vector(vectors[:, 0], unit)
                    partners = reduction.partner_states(state[:, None], np.array([2]))
                    np.testing.assert_allclose(partners.T @ partners, np.eye(2) * (state @ state), atol=1e-12)
                    residual = hamiltonian @ partners - values[0] * partners
                    self.assertLess(float(np.max(np.abs(residual))), 1e-9)


class PeriodicUnitSpectrumTests(unittest.TestCase):
    def test_rocksalt_and_diamond_cells(self):
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
            ("rocksalt", rocksalt, {"Na": 1.0, "Cl": 2.5}, "off", ("once",)),
            ("diamond half", diamond, {"C": 2.0}, "half", ("once", "split")),
            ("diamond auto", diamond, {"C": 2.0}, "auto", ("once", "split")),
        )
        for name, atoms, depths, translations, modes in cases:
            grid, hamiltonian = _periodic_model((side,) * 3, 12, 0.5, atoms, depths)
            full = np.linalg.eigvalsh(hamiltonian.toarray())[:40]
            for mode in modes:
                environment = {"PARSEC_ORBITAL_IRREPS": mode, "PARSEC_PERIODIC_TRANSLATIONS": translations}
                with self.subTest(case=name, mode=mode), patch.dict(os.environ, environment):
                    reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
                    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                    if reduction.sector_projected is None:
                        # the units are the sectors of before (Td x fcc, once)
                        self.assertEqual(mode, "once")
                        continue
                    _check_projectors(self, decomposition, hamiltonian)
                    np.testing.assert_allclose(
                        _unit_spectrum(decomposition, hamiltonian, 40), full, rtol=0.0, atol=1e-10
                    )


if __name__ == "__main__":
    unittest.main()
