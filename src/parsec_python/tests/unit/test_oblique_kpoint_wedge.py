"""The k-points of oblique cells on the default routines (2026-11-21).

Three changes, each behind its switch, make the irreducible wedge of a Monkhorst-Pack mesh of scalar orbitals give the
energy of the whole mesh without symmetry on the same grid, and keep the forces in the crystal's group:

* ``PARSEC_SCALAR_KPOINT_WEDGE=auto`` (default; ``grid`` keeps the former group): the wedge keeps the operations of the
  crystal that map the grid, the mesh and the grid's finite-difference operators onto themselves
  (:func:`parsec_python.Grid.kpoints.stencil_preserving_operations`), as non-magnetic spin-orbit coupling does since
  2026-11-19.  The three- and six-fold axes of a hexagonal grid and the three-fold axes of a primitive fcc or bcc grid
  turn a grid axis onto another direction of the stencil and change the discrete Hamiltonian.
* ``PARSEC_KPOINT_FOLD=shortest`` (default; ``centred`` keeps the former points): every point of the mesh is computed
  at its shortest images modulo the reciprocal lattice (:func:`parsec_python.Grid.kpoints.shortest_images`), the first
  Brillouin zone, instead of the point folded into ``(-1/2, 1/2]`` along the reciprocal vectors.  A point on the zone
  boundary with ``m`` equally short images takes all of them with ``1/m`` of its weight: the finite-difference Bloch
  operator of the u gauge is not invariant under ``k -> k + G``, so a choice of one of them would depend on the order
  of a search and break the lattice's symmetry.  The whole mesh of these shared ties is the same-grid reference; the
  wedge of the stencil-preserving operations, whose stars are split into the orbits of exact images, gives it, and its
  time-reversal pairs are exact.
* ``PARSEC_KPOINT_FORCE_GROUP=crystal`` (default; ``wedge`` averages over the wedge's operations): the forces and the
  moved positions of such a wedge are averaged over every operation that maps the grid and the mesh onto themselves
  (``SinglePointInput.force_space_group``), the group of the converged forces.

On a lattice of orthogonal vectors all three keep every bit: all operations keep the stencil, and the folded points
are the shortest ones (a component 1/2 keeps its tie with -1/2 there).  The Fortran-derived routines keep their points
and their group.
"""

from __future__ import annotations

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python.Grid.cell import reciprocal_lattice
from parsec_python.Grid.kpoints import (
    fold_to_centered_zone,
    fold_to_shortest,
    irreducible_monkhorst_pack,
    kpoint_fold_setting,
    kpoint_force_group_setting,
    kpoint_sampling,
    kpoint_sampling_details,
    negation_closed_shortest,
    oblique_reciprocal_lattice,
    scalar_kpoint_wedge_setting,
    shortest_images,
    tie_shared_points,
    time_reversal_reduced,
)
from parsec_python.tests.unit.test_soc_kpoint_wedge import (
    _FCC,
    _HCP,
    _ZERO_SHIFT,
    _Directory,
    _hexagonal,
    _text,
)

_OLD = {"PARSEC_SCALAR_KPOINT_WEDGE": "grid", "PARSEC_KPOINT_FOLD": "centred"}
_SCALAR = dict(extra="", so_psp=".false.")
_BCC = "begin Cell_Shape\n-4.0 4.0 4.0\n4.0 -4.0 4.0\n4.0 4.0 -4.0\nend Cell_Shape\nLattice_Vector_Scale: 1.0"


def _lattices():
    a, c = 5.45, 8.9
    hexagonal = np.array([[a, 0.0, 0.0], [-a / 2, a * 3 ** 0.5 / 2, 0.0], [0.0, 0.0, c]])
    fcc = np.array([[0.0, 3.85, 3.85], [3.85, 0.0, 3.85], [3.85, 3.85, 0.0]])
    bcc = np.array([[-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, -4.0]])
    monoclinic = np.array([[6.0, 0.0, 0.0], [0.0, 7.0, 0.0], [2.5, 0.0, 6.5]])
    # an unreduced basis of the hexagonal lattice: a2' = a2 + 2 a1, a3' = a3 + a1
    skewed = np.array([hexagonal[0], hexagonal[1] + 2 * hexagonal[0], hexagonal[2] + hexagonal[0]])
    return {"hexagonal": hexagonal, "fcc": fcc, "bcc": bcc, "monoclinic": monoclinic, "skewed": skewed}


def _brute_force(fractions, reciprocal, reach=4):
    """(shortest squared lengths, every image within 1e-9 of them) by an exhaustive search."""

    metric = reciprocal @ reciprocal.T
    shifts = np.asarray(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=np.float64)
    moved = np.asarray(fractions, dtype=np.float64)[:, None, :] - shifts[None, :, :]
    lengths = np.einsum("pci,ij,pcj->pc", moved, metric, moved)
    best = lengths.min(axis=1)
    images = [moved[p][lengths[p] <= best[p] + 1.0e-9 * np.max(np.diag(metric))] for p in range(len(fractions))]
    return best, images


def _mesh(dimensions):
    return np.array(list(itertools.product(*(np.arange(n) / n for n in dimensions))), dtype=np.float64)


def _has_partner(points, row, atol=1.0e-12) -> bool:
    """Whether ``-row`` is one of ``points`` or ``row`` is its own partner (``2 row`` a reciprocal vector)."""

    if np.max(np.abs(2.0 * row - np.rint(2.0 * row))) < atol:
        return True
    return float(np.min(np.max(np.abs(points + row[None, :]), axis=1))) < atol


def _same_rows(first, second, atol=1.0e-12) -> bool:
    first, second = np.atleast_2d(first), np.atleast_2d(second)
    if first.shape != second.shape:
        return False
    used = np.zeros(len(second), dtype=bool)
    for row in first:
        distance = np.max(np.abs(second - row[None, :]), axis=1)
        distance[used] = np.inf
        index = int(np.argmin(distance))
        if distance[index] > atol:
            return False
        used[index] = True
    return True


class FoldTests(unittest.TestCase):
    def test_the_shortest_images(self) -> None:
        generator = np.random.default_rng(7)
        points = np.vstack([generator.uniform(-1.5, 1.5, size=(200, 3)),
                            np.array(list(itertools.product((0.0, 0.25, 0.5, 0.75, -0.5), repeat=3)))])
        for name, lattice in _lattices().items():
            with self.subTest(lattice=name):
                reciprocal = reciprocal_lattice(lattice)
                folded = fold_to_shortest(points, reciprocal)
                # the same points modulo the reciprocal lattice
                difference = folded - points
                np.testing.assert_allclose(difference, np.rint(difference), atol=1.0e-12)
                # as short as any image (exhaustive search, also on the unreduced basis)
                metric = reciprocal @ reciprocal.T
                lengths = np.einsum("ki,ij,kj->k", folded, metric, folded)
                np.testing.assert_allclose(lengths, _brute_force(points, reciprocal)[0], rtol=1.0e-9, atol=1.0e-12)
                # folding again changes nothing
                np.testing.assert_array_equal(fold_to_shortest(folded, reciprocal), folded)
                # one point at a time gives the same
                np.testing.assert_array_equal(fold_to_shortest(points[5], reciprocal), folded[5])

    def test_every_shortest_image_of_the_zone_boundary(self) -> None:
        points = np.vstack([_mesh((6, 6, 6)), _mesh((4, 4, 2)), _mesh((3, 3, 3)) - 0.5])
        for name, lattice in _lattices().items():
            with self.subTest(lattice=name):
                reciprocal = reciprocal_lattice(lattice)
                self.assertTrue(oblique_reciprocal_lattice(reciprocal))
                found = shortest_images(points, reciprocal)
                _, expected = _brute_force(points, reciprocal)
                np.testing.assert_array_equal(
                    np.asarray([group[0] for group in found]), fold_to_shortest(points, reciprocal)
                )
                for index, (group, reference) in enumerate(zip(found, expected)):
                    self.assertTrue(_same_rows(group, reference, atol=1.0e-12), (index, group, reference))
                self.assertGreater(sum(group.shape[0] > 1 for group in found), 0)
        # the fcc W point (1/2, 1/4, 3/4) has four shortest images, the fcc L point two, the hexagonal K point three
        fcc = reciprocal_lattice(_lattices()["fcc"])
        hexagonal = reciprocal_lattice(_lattices()["hexagonal"])
        self.assertEqual(shortest_images([[0.5, 0.25, 0.75]], fcc)[0].shape[0], 4)
        self.assertEqual(shortest_images([[0.5, 0.5, 0.5]], fcc)[0].shape[0], 2)
        self.assertEqual(shortest_images([[1 / 3, 1 / 3, 0.0]], hexagonal)[0].shape[0], 3)

    def test_orthogonal_lattices_keep_the_folded_points_bit_for_bit(self) -> None:
        points = np.array(list(itertools.product(np.arange(6) / 6.0, np.arange(4) / 4.0, np.arange(5) / 5.0)))
        rotation = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
        for lattice in (np.diag([6.2, 6.2, 6.2]), np.diag([5.0, 7.5, 9.0]), np.diag([5.0, 7.5, 9.0]) @ rotation.T):
            reciprocal = reciprocal_lattice(lattice)
            self.assertFalse(oblique_reciprocal_lattice(reciprocal))
            np.testing.assert_array_equal(fold_to_shortest(points, reciprocal), fold_to_centered_zone(points))
            found = shortest_images(points, reciprocal)
            self.assertTrue(all(group.shape[0] == 1 for group in found))
            np.testing.assert_array_equal(np.vstack(found), fold_to_centered_zone(points))
            np.testing.assert_array_equal(negation_closed_shortest(points, reciprocal), fold_to_centered_zone(points))
            k_points = fold_to_centered_zone(points) @ reciprocal
            weights = np.full(len(points), 1.0 / len(points))
            shared = tie_shared_points(k_points, weights, lattice)
            np.testing.assert_array_equal(shared[0], k_points)
            np.testing.assert_array_equal(shared[1], weights)

    def test_the_whole_mesh_of_shared_ties_and_its_pairs(self) -> None:
        for name, dimensions in (("fcc", (6, 6, 6)), ("bcc", (6, 6, 6)), ("hexagonal", (4, 4, 2)), ("fcc", (4, 4, 4))):
            with self.subTest(lattice=name, mesh=dimensions):
                lattice = _lattices()[name]
                reciprocal = reciprocal_lattice(lattice)
                points = _mesh(dimensions)
                single = negation_closed_shortest(points, reciprocal)
                # one shortest image of each point, the set closed under k -> -k exactly (a point with -k = k modulo
                # the reciprocal lattice is its own partner: u_-k = u_k^*)
                for row in single:
                    self.assertTrue(_has_partner(single, row))
                metric = reciprocal @ reciprocal.T
                np.testing.assert_allclose(
                    np.einsum("ki,ij,kj->k", single, metric, single), _brute_force(points, reciprocal)[0], atol=1e-12
                )
                weights = np.full(len(points), 1.0 / len(points))
                k_points, shares = tie_shared_points(single @ reciprocal, weights, lattice)
                fractions = k_points @ np.linalg.inv(reciprocal)
                self.assertGreater(len(k_points), len(points))
                self.assertAlmostEqual(float(shares.sum()), 1.0, places=14)
                # every image of every point, each with its share, and nothing else
                _, expected = _brute_force(points, reciprocal)
                self.assertTrue(_same_rows(fractions, np.vstack(expected), atol=1.0e-10))
                # its time-reversal pairs are exact: every point is a kept one or exactly the negative of one
                kept, kept_weights = time_reversal_reduced(k_points, shares, lattice)
                self.assertAlmostEqual(float(kept_weights.sum()), 1.0, places=14)
                kept_fractions = kept @ np.linalg.inv(reciprocal)
                for row in fractions:
                    distance = np.minimum(
                        np.max(np.abs(kept_fractions - row[None, :]), axis=1),
                        np.max(np.abs(kept_fractions + row[None, :]), axis=1),
                    )
                    self.assertLess(float(distance.min()), 1.0e-10)

    def test_time_reversal_pairs_of_one_point_per_class_keep_their_bits(self) -> None:
        # a list with one point per class modulo G: the partner modulo G, the same pairs as before 2026-11-21
        lattice = np.diag([6.2, 6.2, 6.2])
        reciprocal = reciprocal_lattice(lattice)
        k_points = fold_to_centered_zone(_mesh((4, 4, 4))) @ reciprocal
        kept, weights = time_reversal_reduced(k_points, np.full(64, 1.0 / 64.0), lattice)
        self.assertEqual(len(kept), 36)
        self.assertEqual(sorted(set(np.round(weights * 64).astype(int))), [1, 2])

    def test_the_wedge_with_the_shortest_representatives(self) -> None:
        lattice = _lattices()["hexagonal"]
        for rotations in (None, np.array([np.eye(3, dtype=int), -np.eye(3, dtype=int)])):
            folded = irreducible_monkhorst_pack(lattice, (4, 4, 2), (0.0, 0.0, 0.0), rotations)
            shortest = irreducible_monkhorst_pack(lattice, (4, 4, 2), (0.0, 0.0, 0.0), rotations, shortest=True)
            for name in ("counts", "weights", "grid_map", "kinetic_energies"):
                np.testing.assert_array_equal(getattr(shortest, name), getattr(folded, name))
            np.testing.assert_array_equal(
                shortest.fractional, fold_to_shortest(folded.fractional, reciprocal_lattice(lattice))
            )
            self.assertFalse(np.array_equal(shortest.fractional, folded.fractional))
            np.testing.assert_allclose(shortest.k_points, shortest.fractional @ reciprocal_lattice(lattice), atol=1e-12)
            # the stars of exact images: time reversal (and inversion) alone relates no tie of the hexagonal mesh
            split = irreducible_monkhorst_pack(
                lattice, (4, 4, 2), (0.0, 0.0, 0.0), rotations, shortest=True, ties=True
            )
            self.assertGreater(split.count, shortest.count)
            self.assertAlmostEqual(float(split.weights.sum()), 1.0, places=14)
            self.assertAlmostEqual(float(np.sum(split.counts)), 32.0, places=12)
            self.assertTrue(_wedge_is_the_whole_mesh(split, lattice, rotations))
            self.assertFalse(_wedge_is_the_whole_mesh(shortest, lattice, rotations))
        # a wedge whose stars are exact keeps every bit
        cube = np.diag([6.2, 6.2, 6.2])
        plain = irreducible_monkhorst_pack(cube, (4, 4, 4), (0.0, 0.0, 0.0), None, shortest=True)
        kept = irreducible_monkhorst_pack(cube, (4, 4, 4), (0.0, 0.0, 0.0), None, shortest=True, ties=True)
        for name in ("fractional", "k_points", "counts", "weights", "grid_map", "kinetic_energies"):
            np.testing.assert_array_equal(getattr(kept, name), getattr(plain, name))
            self.assertEqual(getattr(kept, name).dtype, getattr(plain, name).dtype)

    def test_switch_values(self) -> None:
        for name, function, values, default in (
            ("PARSEC_KPOINT_FOLD", kpoint_fold_setting, ("shortest", "centred"), "shortest"),
            ("PARSEC_SCALAR_KPOINT_WEDGE", scalar_kpoint_wedge_setting, ("auto", "grid"), "auto"),
            ("PARSEC_KPOINT_FORCE_GROUP", kpoint_force_group_setting, ("crystal", "wedge"), "crystal"),
        ):
            with self.subTest(name=name):
                with mock.patch.dict(os.environ, {name: ""}):
                    self.assertEqual(function(), default)
                for value in values:
                    with mock.patch.dict(os.environ, {name: f" {value.upper()} "}):
                        self.assertEqual(function(), value)
                with mock.patch.dict(os.environ, {name: "nonsense"}):
                    with self.assertRaisesRegex(ValueError, name):
                        function()


def _wedge_is_the_whole_mesh(sampling, lattice, rotations=None, time_reversal=True) -> bool:
    """Whether the wedge ``sampling``, each representative's orbit ``+-G f`` sharing its weight, is the whole mesh of
    shared ties (every shortest image of every point with its share, :func:`tie_shared_points`) as a weighted set:
    then the wedge computes exactly the k-points of that mesh."""

    matrices = [np.eye(3)] if rotations is None else [np.asarray(r, dtype=np.float64) for r in rotations]
    points, weights = [], []
    for fraction, weight in zip(np.asarray(sampling.fractional), np.asarray(sampling.weights)):
        images = [matrix @ fraction for matrix in matrices]
        if time_reversal:
            images += [-image for image in images]
        distinct = []
        for image in images:
            if not any(np.max(np.abs(image - other)) < 1.0e-8 for other in distinct):
                distinct.append(image)
        points += distinct
        weights += [weight / len(distinct)] * len(distinct)
    reciprocal = reciprocal_lattice(lattice)
    full = sampling.full_grid_fractional().reshape(-1, 3)
    groups = shortest_images(full, reciprocal)
    reference = np.vstack(groups)
    shares = np.concatenate([np.full(group.shape[0], 1.0 / (len(full) * group.shape[0])) for group in groups])
    if len(points) != len(reference):
        return False
    used = np.zeros(len(reference), dtype=bool)
    for point, weight in zip(points, weights):
        distance = np.max(np.abs(reference - point[None, :]), axis=1)
        distance[used] = np.inf
        index = int(np.argmin(distance))
        if distance[index] > 1.0e-8 or abs(weight - shares[index]) > 1.0e-12:
            return False
        used[index] = True
    return True


def _problem_wedge_is_the_whole_mesh(problem) -> bool:
    details = kpoint_sampling_details(problem)
    rotations = None if problem.space_group is None else problem.space_group.reciprocal_rotations
    return _wedge_is_the_whole_mesh(details, problem.periodic_cell.lattice_vectors, rotations)


class ParserTests(_Directory):
    def test_the_scalar_wedge_of_oblique_cells(self) -> None:
        # (cell, mesh, operations and points of the new defaults, of the former group)
        for cell, mesh, new, old in (
            ("hcp", "4 4 2", (8, 14), (24, 8)),
            ("fcc", "4 4 4", (12, 14), (48, 8)),
            ("fcc", "6 6 6", (12, 34), (48, 16)),
            ("fcc", "8 8 8", (12, 71), (48, 29)),
            ("bcc", "6 6 6", (12, 34), (48, 16)),
            ("hcp", "6 6 4", (8, 42), (24, 21)),
        ):
            with self.subTest(cell=cell, mesh=mesh):
                text = _text(cell="hcp" if cell == "hcp" else "fcc", mesh=mesh, **_SCALAR)
                if cell == "bcc":
                    text = text.replace(_FCC, _BCC)
                translation = self.parse(text)
                problem = translation.problem
                self.assertFalse(translation.routines.fortran)
                self.assertTrue(problem.kpoint_shortest_representatives)
                self.assertEqual((problem.space_group.order, len(kpoint_sampling(problem)[0])), new)
                self.assertIn("PARSEC_SCALAR_KPOINT_WEDGE=grid keeps them", " ".join(translation.warnings))
                self.assertTrue(problem.kpoint_tie_images)
                self.assertIn("lie on the boundary of the first Brillouin zone", " ".join(translation.warnings))
                # the wedge computes exactly the points of the whole mesh of shared ties
                self.assertTrue(_problem_wedge_is_the_whole_mesh(problem))
                before = self.parse(text, _OLD).problem
                self.assertFalse(before.kpoint_shortest_representatives)
                self.assertEqual((before.space_group.order, len(kpoint_sampling(before)[0])), old)
                # its stars reach the same points by operations that change the finite-difference operators: those
                # that keep them do not reach the whole mesh from its representatives
                self.assertFalse(
                    _wedge_is_the_whole_mesh(
                        kpoint_sampling_details(before),
                        problem.periodic_cell.lattice_vectors,
                        problem.space_group.reciprocal_rotations,
                    )
                )
                # the stencil-preserving operations alone, with the folded points, are not enough
                stencil = self.parse(text, {"PARSEC_KPOINT_FOLD": "centred"}).problem
                self.assertEqual(stencil.space_group.order, new[0])
                self.assertFalse(_problem_wedge_is_the_whole_mesh(stencil))

    def test_the_whole_mesh_of_moving_atoms(self) -> None:
        for extra, environment in (
            ("Skip_Force: .false.", {"PARSEC_KPOINT_FORCE_WEDGE": "off"}),
            ("Molecular_Dynamics: .true.\nStep_Num: 2", {}),
        ):
            with self.subTest(extra=extra):
                new = self.parse(_text(cell="hcp", extra=extra, so_psp=".false."), environment).problem
                old = self.parse(_text(cell="hcp", extra=extra, so_psp=".false."), {**_OLD, **environment}).problem
                self.assertTrue(new.kpoint_time_reversal)
                self.assertTrue(new.kpoint_tie_images)
                self.assertFalse(old.kpoint_tie_images)
                reciprocal = reciprocal_lattice(new.periodic_cell.lattice_vectors)
                new_fractions = new.kpoints @ np.linalg.inv(reciprocal)
                old_fractions = old.kpoints @ np.linalg.inv(reciprocal)
                # one shortest image of each point of the mesh, in the order of the mesh
                for row, group in zip(new_fractions, shortest_images(old_fractions, reciprocal)):
                    self.assertTrue(np.any(np.max(np.abs(group - row[None, :]), axis=1) < 1.0e-12))
                self.assertFalse(np.allclose(new_fractions, old_fractions))
                # the run takes every shortest image with its share, as its exact time-reversal pairs
                k_points, weights = kpoint_sampling(new)
                self.assertEqual(len(k_points), 29)
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
                self.assertEqual(len(kpoint_sampling(old)[0]), 20)

    def test_the_dos_and_output_gw_keep_one_point_per_class(self) -> None:
        shared = self.parse(_text(cell="fcc", **_SCALAR)).problem
        self.assertTrue(shared.kpoint_tie_images)
        # the tetrahedra and BerkeleyGW's unfolding need one point per class: those inputs run on the Fortran-derived
        # routines (also with Use_Fortran_Routines: .false.), whose points and stars do not change
        for extra in ("Create_DOS: .true.", "Output_GW: .true."):
            with self.subTest(extra=extra):
                translation = self.parse(
                    _text(cell="fcc", extra=extra + "\nUse_Fortran_Routines: .false.", so_psp=".false.")
                )
                self.assertTrue(translation.routines.fortran)
                problem = translation.problem
                self.assertFalse(problem.kpoint_tie_images)
                self.assertFalse(problem.kpoint_shortest_representatives)
                details = kpoint_sampling_details(problem)
                self.assertEqual(details.counts.dtype, np.int64)
                self.assertEqual(int(np.sum(details.counts)), 64)

    def test_a_hybrid_keeps_one_point_per_class(self) -> None:
        problem = self.parse(_text(cell="hcp", xc="PBE0", **_SCALAR), {"PARSEC_HYBRID_KPOINT_WEDGE": "off"}).problem
        self.assertTrue(problem.scf.is_hybrid)
        self.assertFalse(problem.kpoint_tie_images)
        self.assertEqual(len(problem.kpoints), 32)
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        fractions = problem.kpoints @ np.linalg.inv(reciprocal)
        for row in fractions:
            self.assertTrue(_has_partner(fractions, row))

    def test_orthogonal_cells_keep_every_bit(self) -> None:
        for extra, environment in (
            ("", {}),
            ("Skip_Force: .false.", {}),
            ("Skip_Force: .false.", {"PARSEC_KPOINT_FORCE_WEDGE": "off"}),
            ("Molecular_Dynamics: .true.\nStep_Num: 2", {}),
        ):
            with self.subTest(extra=extra, environment=environment):
                text = _text(cell="cube", extra=extra, so_psp=".false.")
                new = self.parse(text, environment).problem
                old = self.parse(text, {**_OLD, **environment}).problem
                for name in ("kpoints", "kpoint_weights"):
                    first, second = getattr(new, name), getattr(old, name)
                    self.assertEqual(first is None, second is None)
                    if first is not None:
                        np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(kpoint_sampling(new)[0], kpoint_sampling(old)[0])
                np.testing.assert_array_equal(kpoint_sampling(new)[1], kpoint_sampling(old)[1])
                self.assertEqual(new.space_group is None, old.space_group is None)
                self.assertFalse(new.kpoint_tie_images)
                self.assertIsNone(new.force_space_group)
                self.assertEqual(
                    self.parse(text, environment).warnings, self.parse(text, {**_OLD, **environment}).warnings
                )
                if new.space_group is not None:
                    self.assertEqual(new.space_group.rotations, old.space_group.rotations)
                    self.assertEqual(new.space_group.translations, old.space_group.translations)

    def test_spin_orbit_coupling_takes_the_shortest_representatives(self) -> None:
        new = self.parse(_text(cell="fcc")).problem
        old = self.parse(_text(cell="fcc"), {"PARSEC_KPOINT_FOLD": "centred"}).problem
        self.assertEqual(new.space_group.order, old.space_group.order)
        self.assertTrue(_problem_wedge_is_the_whole_mesh(new))
        self.assertFalse(_problem_wedge_is_the_whole_mesh(old))
        # PARSEC_SOC_KPOINT_WEDGE=grid keeps the former group of scalar orbitals
        grid = self.parse(_text(cell="fcc"), {"PARSEC_SOC_KPOINT_WEDGE": "grid"}).problem
        self.assertEqual(grid.space_group.order, 48)

    def test_the_forces_keep_the_group_of_the_grid_and_the_mesh(self) -> None:
        from parsec_python.Forces.periodic import periodic_force_operations
        from parsec_python.Forces.symmetry import symmetrize_forces

        # hcp Au2, the second atom moved along c: the atoms stay on the three-fold axes
        coordinates = _hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.78)])
        for extra in ("Skip_Force: .false.", "Minimization: bfgs\nMovement_Num: 2"):
            with self.subTest(extra=extra):
                text = _text(cell="hcp", extra=extra, so_psp=".false.")
                text = text.replace(_hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)]), coordinates)
                translation = self.parse(text)
                problem = translation.problem
                self.assertEqual(problem.space_group.order, 2)
                self.assertEqual(problem.force_space_group.order, 6)
                self.assertIn("PARSEC_KPOINT_FORCE_GROUP=wedge", " ".join(translation.warnings))
                operations = periodic_force_operations(SimpleNamespace(input=problem, atoms=problem.atoms))
                self.assertEqual(len(operations), 6)
                forces = symmetrize_forces(np.random.default_rng(5).normal(size=(2, 3)), operations)
                np.testing.assert_allclose(forces[:, :2], 0.0, atol=1.0e-12)
                wedge = self.parse(text, {"PARSEC_KPOINT_FORCE_GROUP": "wedge"}).problem
                self.assertIsNone(wedge.force_space_group)
                self.assertEqual(wedge.space_group.order, 2)
                self.assertEqual(len(periodic_force_operations(SimpleNamespace(input=wedge, atoms=wedge.atoms))), 2)
                former = self.parse(text, _OLD).problem
                self.assertIsNone(former.force_space_group)
                self.assertEqual(former.space_group.order, 6)

    def test_the_whole_mesh_of_a_cell_whose_operations_all_turn_the_stencil(self) -> None:
        from parsec_python.Forces.periodic import periodic_force_operations

        # C3 alone, whose rotations turn the grid axes of the hexagonal cell: the whole mesh (its time-reversal pairs)
        # runs, and its forces are averaged over the 3 operations
        for extra in ("Skip_Force: .false.", "Minimization: bfgs\nMovement_Num: 2"):
            with self.subTest(extra=extra):
                translation = self.parse(_text(cell="c3", extra=extra, so_psp=".false."))
                problem = translation.problem
                self.assertIsNone(problem.space_group)
                self.assertTrue(problem.kpoint_time_reversal)
                self.assertEqual(problem.force_space_group.order, 3)
                self.assertIn("which the k-point sum of the whole mesh does not use", " ".join(translation.warnings))
                operations = periodic_force_operations(SimpleNamespace(input=problem, atoms=problem.atoms))
                self.assertEqual(len(operations), 3)
                wedge = self.parse(
                    _text(cell="c3", extra=extra, so_psp=".false."), {"PARSEC_KPOINT_FORCE_GROUP": "wedge"}
                ).problem
                self.assertIsNone(wedge.force_space_group)
                np.testing.assert_array_equal(kpoint_sampling(wedge)[0], kpoint_sampling(problem)[0])

    def test_the_fortran_routines_keep_their_points(self) -> None:
        text = _text(cell="fcc", shift=_ZERO_SHIFT, **_SCALAR)
        new = self.parse(text)
        old = self.parse(text, _OLD)
        self.assertTrue(new.routines.fortran)
        self.assertFalse(new.problem.kpoint_shortest_representatives)
        self.assertFalse(new.problem.kpoint_tie_images)
        np.testing.assert_array_equal(kpoint_sampling(new.problem)[0], kpoint_sampling(old.problem)[0])
        self.assertEqual(new.problem.space_group.order, 48)

    def test_every_input_checks_the_switches(self) -> None:
        for name in ("PARSEC_KPOINT_FOLD", "PARSEC_SCALAR_KPOINT_WEDGE", "PARSEC_KPOINT_FORCE_GROUP"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(Exception, name):
                    self.parse(_text(cell="cube", **_SCALAR), {name: "on"})

    def test_a_slab_of_the_default_routines(self) -> None:
        text = _text(cell="hcp", mesh="4 4", extra="Use_Fortran_Routines: .false.", so_psp=".false.")
        text = text.replace("Boundary_Conditions: bulk", "Boundary_Conditions: slab\nBoundary_Sphere_Radius: 6.0")
        lattice = _HCP.split("\n")
        text = text.replace(_HCP, "\n".join([lattice[0], lattice[1], lattice[2], *lattice[4:]]))
        text = text.replace(_hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)]),
                            "  0.0 0.0 0.0\n  0.0 3.1465 0.0")
        new = self.parse(text)
        old = self.parse(text, _OLD).problem
        self.assertFalse(new.routines.fortran)
        new = new.problem
        self.assertIsNone(new.space_group)
        self.assertEqual(len(new.kpoints), 16)
        np.testing.assert_allclose(new.kpoints[:, 2], 0.0, atol=1.0e-14)
        reciprocal = reciprocal_lattice(new.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(
            new.kpoints @ np.linalg.inv(reciprocal),
            negation_closed_shortest(old.kpoints @ np.linalg.inv(reciprocal), reciprocal),
            atol=1.0e-12,
        )
        self.assertFalse(np.allclose(new.kpoints, old.kpoints))
        # the run: the M points (1/2, 0), (0, 1/2), (1/2, 1/2) with both their shortest images
        self.assertTrue(new.kpoint_tie_images)
        k_points, weights = kpoint_sampling(new)
        self.assertEqual(len(k_points), 16 + 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        np.testing.assert_allclose(k_points[:, 2], 0.0, atol=1.0e-14)


def _archive(directory):
    from parsec_python.tests.archives import read_archive

    return read_archive(directory / "parsec_python_results.npz")


class RunTests(_Directory):
    """A coarse hcp Au2 cell on the CPU: the wedge of the new defaults against the whole mesh without symmetry."""

    def run_cli(self, text: str, environment=None) -> dict:
        from parsec_python.cli import main as cli_main

        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            code = cli_main([str(path), "--quiet"])
        log = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertEqual(code, 0, log[-2000:])
        archive = _archive(self.directory)
        self.assertTrue(bool(archive["converged"]))
        return {"energy": float(archive["energy_total_ry"]), "k": int(archive["n_kpoints"]),
                "forces": np.asarray(archive["forces_total"])}

    def test_the_wedge_is_the_whole_mesh(self) -> None:
        from parsec_python.Forces.periodic import periodic_force_operations
        from parsec_python.Forces.symmetry import symmetrize_forces

        coordinates = _hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.80)])
        text = _text(cell="hcp", mesh="4 4 1", spacing=0.6, extra="Skip_Force: .false.", so_psp=".false.")
        text = text.replace(_hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)]), coordinates)
        text = text.replace("Convergence_Criterion: 1.0e-6", "Convergence_Criterion: 1.0e-8")
        text = text.replace("States_Num: 32", "States_Num: 20")
        wedge = self.run_cli(text, {"PARSEC_KPOINT_FORCE_GROUP": "wedge"})
        # the whole mesh of the same points without symmetry: the time-reversal pairs of the forces (every shortest
        # image of the points on the zone boundary, whose partners are exactly -k)
        whole = self.run_cli(text, {"PARSEC_KPOINT_FORCE_WEDGE": "off"})
        self.assertLess(wedge["k"], whole["k"])
        self.assertAlmostEqual(wedge["energy"], whole["energy"], delta=1.0e-6)
        np.testing.assert_allclose(wedge["forces"], whole["forces"], atol=1.0e-5)
        # the default: the same k-point sum, the forces averaged over the 6 operations of the grid and the mesh
        default = self.run_cli(text)
        self.assertEqual(default["k"], wedge["k"])
        self.assertAlmostEqual(default["energy"], wedge["energy"], delta=1.0e-8)
        problem = self.parse(text).problem
        operations = periodic_force_operations(SimpleNamespace(input=problem, atoms=problem.atoms))
        self.assertEqual(len(operations), 6)
        np.testing.assert_allclose(default["forces"], symmetrize_forces(whole["forces"], operations), atol=1.0e-5)
        np.testing.assert_allclose(default["forces"][:, :2], 0.0, atol=1.0e-12)

    def test_the_whole_mesh_forces_of_a_c3_cell_keep_its_group(self) -> None:
        from parsec_python.Forces.periodic import periodic_force_operations
        from parsec_python.Forces.symmetry import symmetrize_forces

        # C3 alone (every rotation turns the stencil): the whole mesh as its pairs, the forces averaged over C3
        text = _text(cell="c3", mesh="2 2 1", spacing=0.55, extra="Skip_Force: .false.", so_psp=".false.")
        text = text.replace("States_Num: 100", "States_Num: 50")
        wedge = self.run_cli(text, {"PARSEC_KPOINT_FORCE_GROUP": "wedge"})
        default = self.run_cli(text)
        self.assertEqual(default["k"], wedge["k"])
        self.assertAlmostEqual(default["energy"], wedge["energy"], delta=1.0e-10)
        translation = self.parse(text)
        problem = translation.problem
        self.assertFalse(translation.routines.fortran)
        self.assertIsNone(problem.space_group)
        self.assertEqual(problem.force_space_group.order, 3)
        operations = periodic_force_operations(SimpleNamespace(input=problem, atoms=problem.atoms))
        self.assertEqual(len(operations), 3)
        np.testing.assert_allclose(default["forces"], symmetrize_forces(wedge["forces"], operations), atol=1.0e-9)
        self.assertGreater(float(np.max(np.abs(wedge["forces"] - default["forces"]))), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
