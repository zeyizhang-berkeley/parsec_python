"""The mirror pairs of the image sources of a hybrid's k-point wedge (``PARSEC_HYBRID_IMAGE_PAIRS``).

The pair of the target ``(t, n)`` with the image source ``(i, m)`` (``P_g u_rm``) and the pair of ``(r, m)`` with the
mirror source ``(j, n)`` (``P_(g^-1) u_tn``) are served by one transform where the pair potentials keep the operation
(``V_xc.bloch_exchange.MeshStars.mirrors``, ``image_pair_relation_holds``).  The relation is algebraic (a permutation
of the grid and an isotropic kernel), so it holds for any orbitals: the checks use random ones.
"""

from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock
import warnings

import numpy as np


def _cube_wedge(mesh: str = "2 2 2", shift_block: bool = False):
    """The silicon cube of 8 atoms with the wedge of its mesh (``PARSEC_HYBRID_KPOINT_WEDGE=on``): the system, its
    k-points and weights, the box and the stars."""

    from parsec_python.acceleration.tests.test_kpoint_hybrid_sectors import _silicon_mesh
    from parsec_python.Grid.kpoints import kpoint_sampling
    from parsec_python.SCF.hybrid_periodic import hybrid_mesh_stars
    from parsec_python.SCF.pbc import prepare_periodic_single_point
    from parsec_python.V_xc.bloch_exchange import GridBox

    with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"}):
        directory, translation = _silicon_mesh("pbe0", "on", mesh=mesh, shift_block=shift_block)
        problem = translation.problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
    shutil.rmtree(directory, True)
    k_points, weights = kpoint_sampling(problem)
    box = GridBox(system.grid)
    return system, k_points, weights, box, hybrid_mesh_stars(system, k_points, weights, box)


def _random_set(k_points, weights, box, stars, seed: int = 1):
    from parsec_python.V_xc.bloch_exchange import bloch_orbital_set

    generator = np.random.default_rng(seed)
    blocks, occupations = [], []
    for _ in k_points:
        block = generator.standard_normal((box.size, 6)) + 1j * generator.standard_normal((box.size, 6))
        blocks.append(np.linalg.qr(block)[0])
        # integer and fractional occupations, two empty states
        occupations.append(np.array([1.0, 1.0, 0.7, 0.3, 0.0, 0.0]))
    return bloch_orbital_set(k_points, weights, blocks, occupations, 2.0, box, stars)


class MirrorTests(unittest.TestCase):
    def test_the_mirrors_of_the_stars(self) -> None:
        _system, k_points, _weights, _box, stars = _cube_wedge("2 2 2")
        self.assertEqual(len(stars.mirrors), 8)
        for (target, index), mirror in stars.mirrors.items():
            rep, operation, conjugate, _label = stars.sources[index]
            mirror_rep, mirror_operation, mirror_conjugate, _mirror_label = stars.sources[mirror]
            # an involution between two image sources of the same conjugation, the operations inverse
            self.assertGreaterEqual(operation, 0)
            self.assertEqual(mirror_rep, target)
            self.assertEqual(mirror_conjugate, conjugate)
            self.assertEqual(stars.mirrors[(rep, mirror)], index)
            composed = stars.gathers[operation][stars.gathers[mirror_operation]]
            np.testing.assert_array_equal(composed, np.arange(composed.size))
        # the Gamma point and R (the little group of the target is the whole group) have none
        targets = {target for target, _index in stars.mirrors}
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(k_points), 4)

    def test_the_relation_of_the_pair_potentials(self) -> None:
        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.SCF.hybrid_periodic import periodic_pair_potentials
        from parsec_python.V_xc.bloch_exchange import image_pair_relation_holds, image_pair_test_density

        system, k_points, _weights, box, stars = _cube_wedge("2 2 2")
        potentials = periodic_pair_potentials(system, hybrid_definition(system.input.scf))
        density = image_pair_test_density(box.size)
        for (target, index), mirror in stars.mirrors.items():
            rep, _operation, conjugate, label = stars.sources[index]
            _mirror_rep, mirror_operation, _c, mirror_label = stars.sources[mirror]
            gather = stars.gathers[mirror_operation]
            first = k_points[target] - label
            second = k_points[rep] - mirror_label
            self.assertTrue(image_pair_relation_holds(potentials.solve, density, first, second, gather, conjugate))
            # another wave vector (a reciprocal lattice vector added), another permutation: no
            self.assertFalse(
                image_pair_relation_holds(
                    potentials.solve, density, first, second + np.asarray(box.reciprocal)[0], gather, conjugate
                )
            )
            self.assertFalse(
                image_pair_relation_holds(
                    potentials.solve, density, first, second, np.roll(np.arange(box.size), 1), conjugate
                )
            )

    def test_the_switch(self) -> None:
        from parsec_python.V_xc.bloch_exchange import image_pairs_setting

        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_IMAGE_PAIRS": ""}):
            self.assertTrue(image_pairs_setting())
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_IMAGE_PAIRS": "off"}):
            self.assertFalse(image_pairs_setting())
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_IMAGE_PAIRS": "auto"}):
            with self.assertRaises(ValueError):
                image_pairs_setting()


class HostExchangeTests(unittest.TestCase):
    def check(self, mesh: str, shift_block: bool = False) -> None:
        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.SCF.hybrid_periodic import periodic_pair_potentials
        from parsec_python.V_xc.bloch_exchange import apply_bloch_exchange

        system, k_points, weights, box, stars = _cube_wedge(mesh, shift_block)
        self.assertTrue(stars.mirrors)
        orbital_set = _random_set(k_points, weights, box, stars)
        potentials = periodic_pair_potentials(system, hybrid_definition(system.input.scf))
        for columns in ("all", "occupied"):
            off = apply_bloch_exchange(orbital_set, potentials, columns=columns, image_pairs=False, batch_size=5)
            on = apply_bloch_exchange(orbital_set, potentials, columns=columns, image_pairs=True, batch_size=5)
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_IMAGE_PAIRS": "off"}):
                read = apply_bloch_exchange(orbital_set, potentials, columns=columns, batch_size=5)
            scale = max(float(np.max(np.abs(block))) for block in off.applied)
            for a, b, c in zip(on.applied, off.applied, read.applied):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=1.0e-13 * scale)
                # off is the former loop, bit for bit
                np.testing.assert_array_equal(b, c)
            self.assertAlmostEqual(on.energy, off.energy, delta=1.0e-13 * abs(off.energy))
            self.assertLess(on.pair_solves, off.pair_solves)
            # and the pairs of every source without the pair symmetry at all
            plain = apply_bloch_exchange(orbital_set, potentials, columns=columns, pair_symmetry=False, batch_size=5)
            for a, b in zip(on.applied, plain.applied):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=1.0e-13 * scale)
        # the occupied columns, then the empty ones
        known = apply_bloch_exchange(orbital_set, potentials, columns="occupied", image_pairs=True)
        done = apply_bloch_exchange(orbital_set, potentials, columns="all", known=known, image_pairs=True)
        full = apply_bloch_exchange(orbital_set, potentials, columns="all", image_pairs=False)
        for a, b in zip(done.applied, full.applied):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=1.0e-13 * scale)

    def test_the_mesh_2x2x2(self) -> None:
        self.check("2 2 2")

    def test_the_mesh_4x4x4(self) -> None:
        # operations that are their own inverse: blocks that are their own mirror (the triangle m >= n)
        _system, _k, _w, _box, stars = _cube_wedge("4 4 4")
        self.assertTrue(any(key == (stars.sources[mirror][0], mirror) for key, mirror in stars.mirrors.items()))
        self.check("4 4 4")

    def test_the_fortran_derived_routines(self) -> None:
        self.check("2 2 2", shift_block=True)


class SummaryLineTests(unittest.TestCase):
    """The ``K-points`` line of the input summary (``--dry-run``) of Fortran's grid says how a hybrid's mesh is
    reduced: by time reversal alone (the density average of the Fortran-derived routines keeps the whole space group,
    the mesh does not), its wedge under ``on``, the wedge of the Fock loop of an insulating start under ``auto``."""

    def line(self, functional: str, wedge: str, shift_block: bool = True) -> str:
        from parsec_python.acceleration.tests.test_kpoint_hybrid_sectors import _silicon_mesh
        from parsec_python.Input.parsec_input import summarize_translation

        directory, translation = _silicon_mesh(functional, wedge, shift_block=shift_block)
        shutil.rmtree(directory, True)
        found = [line for line in summarize_translation(translation).splitlines() if line.startswith("K-points:")]
        self.assertEqual(len(found), 1, found)
        return found[0]

    def test_the_lines(self) -> None:
        semilocal = "(PARSEC's grid, reduced to the irreducible wedge by 48 space-group operations and time reversal)"
        for wedge in ("off", "auto", "on"):
            self.assertTrue(self.line("pbe", wedge).endswith(semilocal))
            self.assertTrue(self.line("pbe", wedge, shift_block=False).endswith(semilocal))
        self.assertIn("reduced by time reversal alone: a hybrid functional's exchange", self.line("pbe0", "off"))
        auto = self.line("pbe0", "auto")
        self.assertIn("reduced by time reversal alone for the semilocal start", auto)
        self.assertIn("wedge of time reversal and 48 space-group operations", auto)
        for shift_block in (True, False):
            on = self.line("pbe0", "on", shift_block=shift_block)
            self.assertIn("by time reversal and the 48 space-group operations that keep the mesh", on)
        for wedge in ("off", "auto", "on"):
            self.assertNotIn("by 48 space-group operations and time reversal", self.line("pbe0", wedge))


if __name__ == "__main__":
    unittest.main()
