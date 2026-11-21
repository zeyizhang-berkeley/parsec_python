"""First counts of weighted sectors from their shares of the valence orbitals (``PARSEC_WEIGHTED_FIRST_SHARE``).

CPU only: the switch, the characters of the spherical harmonics, the counts of the valence orbitals of methane in
the sectors of D2d (Td) against the ranks of those orbitals projected onto each sector, the shares and the first
counts that take them.
"""

from __future__ import annotations

import math
import os
import unittest
from unittest import mock

import numpy as np


def _methane():
    from parsec_python.models import Atom

    a = 1.2
    return [Atom("C", (0.0, 0.0, 0.0))] + [
        Atom("H", tuple(a * np.array(signs, dtype=float)))
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    ]


class WeightedFirstShareTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.Symmetry.sector_shares import weighted_first_share_setting

        for value, expected in (("", "even"), ("even", "even"), ("ORBITALS", "orbitals")):
            with mock.patch.dict(os.environ, {"PARSEC_WEIGHTED_FIRST_SHARE": value}):
                self.assertEqual(weighted_first_share_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_WEIGHTED_FIRST_SHARE": "auto"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_WEIGHTED_FIRST_SHARE"):
                weighted_first_share_setting()

    def test_characters_of_the_harmonics(self) -> None:
        from parsec_python.acceleration.Symmetry.sector_shares import angular_character

        identity = np.eye(3)
        inversion = -np.eye(3)
        c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        c3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        c2 = np.diag([-1, -1, 1])
        mirror = np.diag([1, 1, -1])
        s4 = mirror @ c4
        expected = {
            0: (1, 1, 1, 1, 1, 1, 1),
            1: (3, -3, 1, 0, -1, 1, -1),
            2: (5, 5, -1, -1, 1, 1, -1),
            3: (7, -7, -1, 1, -1, 1, 1),
        }
        for l, values in expected.items():
            found = [angular_character(l, m) for m in (identity, inversion, c4, c3, c2, mirror, s4)]
            np.testing.assert_allclose(found, values, atol=1e-12, err_msg=f"l={l}")

    def test_counts_of_methane_equal_the_ranks_of_the_projected_orbitals(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import (
            PointGroupReduction,
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.acceleration.Symmetry.sector_shares import valence_orbital_sector_counts
        from parsec_python.models import GridSettings

        atoms = _methane()
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=4, shift=(0.5,) * 3))
        reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        self.assertIsInstance(reduction, PointGroupReduction)
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        weights = np.asarray(decomposition.sector_weights)
        counts = valence_orbital_sector_counts(decomposition, atoms, {"C": (0, 1), "H": (0,)}, grid)
        self.assertIsNotNone(counts)
        self.assertEqual(int(np.sum(weights * counts)), 8)
        points = np.asarray(grid.physical_coordinates(grid.integer_coordinates))
        functions = []
        for atom in atoms:
            offset = points - np.asarray(atom.position)
            gaussian = np.exp(-np.sum(offset**2, axis=1) / 1.5)
            functions.append(gaussian)
            if atom.symbol == "C":
                functions += [offset[:, axis] * gaussian for axis in range(3)]
        scale = max(float(np.linalg.norm(f)) for f in functions)
        for sector in range(decomposition.representation_count):
            reduced = np.column_stack([decomposition.reduce_vector(f, sector) for f in functions])
            singular = np.linalg.svd(reduced, compute_uv=False)
            rank = int(np.sum(singular > 1e-8 * scale))
            self.assertEqual(rank, int(counts[sector]), sector)
        # without channels of a species, or with an atom off its images, no counts
        self.assertIsNone(valence_orbital_sector_counts(decomposition, atoms, {"C": (0, 1)}, grid))
        # attached under the setting orbitals; the same count in every sector (the shares of a bulk cell) attaches
        # none, the even share stays
        from types import SimpleNamespace

        from parsec_python.acceleration.Symmetry import sector_shares

        tables = {symbol: SimpleNamespace(channel_occupations=dict.fromkeys(channels, 1.0))
                  for symbol, channels in (("C", (0, 1)), ("H", (0,)))}
        with mock.patch.dict(os.environ, {"PARSEC_WEIGHTED_FIRST_SHARE": "orbitals"}):
            found = sector_shares.attach_valence_orbital_counts(decomposition, atoms, tables, grid)
            np.testing.assert_array_equal(found, counts)
            np.testing.assert_array_equal(decomposition.valence_orbital_counts, counts)
            with mock.patch.object(sector_shares, "valence_orbital_sector_counts",
                                   return_value=np.full(counts.shape, 2, dtype=np.int64)):
                self.assertIsNone(sector_shares.attach_valence_orbital_counts(decomposition, atoms, tables, grid))
            self.assertFalse(hasattr(decomposition, "valence_orbital_counts"))
        for setting in ("", "even"):
            with mock.patch.dict(os.environ, {"PARSEC_WEIGHTED_FIRST_SHARE": setting}):
                self.assertIsNone(sector_shares.attach_valence_orbital_counts(decomposition, atoms, tables, grid))

    def test_shares_and_first_counts(self) -> None:
        from parsec_python.acceleration.Eigensolvers.symmetry import first_sector_count
        from parsec_python.acceleration.Symmetry.sector_shares import predicted_level_shares

        weights = np.array([1, 1, 3, 3])
        # the valence orbitals of the nanodiamond k06 (C795H300) in the sectors of D2d
        counts = np.array([476, 394, 394, 476])
        self.assertEqual(int(np.sum(weights * counts)), 3480)
        shares = predicted_level_shares(counts, weights, 1746)
        np.testing.assert_allclose(np.sum(weights * shares), 1.0, rtol=0, atol=1e-14)
        # below the orbitals: their shares; the levels beyond them evenly
        np.testing.assert_allclose(shares * 8, counts / 3480 * 8, rtol=0, atol=1e-14)
        beyond = predicted_level_shares(counts, weights, 4000)
        np.testing.assert_allclose(beyond * 4000, counts + 520 / 8, rtol=0, atol=1e-9)
        self.assertIsNone(predicted_level_shares(np.zeros(4), weights, 10))
        # the first counts: the share with 3 % (none for the count without margin), the buffer, the rows
        for sector in range(4):
            count = first_sector_count(10_000, weights, 1746, 8, share=float(shares[sector]))
            self.assertEqual(count, math.ceil(1746 * shares[sector] * 1.03) + 8)
            low = first_sector_count(10_000, weights, 1746, 8, 0.0, share=float(shares[sector]))
            self.assertEqual(low, math.ceil(1746 * shares[sector]) + 8)
        self.assertEqual(first_sector_count(100, weights, 1746, 8, share=float(shares[0])), 100)
        # the even share of before where no share is given (and for sectors without weights)
        self.assertEqual(first_sector_count(10_000, weights, 1746, 8), -(-int(1746 * 1.1) // 8) + 8)
        self.assertEqual(first_sector_count(10_000, np.ones(4), 1746, 8, share=0.3),
                         max(-(-1746 // 4), 1746 // 4 + 8))
        # fewer columns in all than the even share with its margin, at least the requested states
        predicted = sum(int(w) * first_sector_count(10_000, weights, 1746, 0, share=float(s))
                        for w, s in zip(weights, shares))
        even = sum(int(w) * first_sector_count(10_000, weights, 1746, 0) for w in weights)
        self.assertGreaterEqual(predicted, 1746)
        self.assertLess(predicted, even)


if __name__ == "__main__":
    unittest.main()
