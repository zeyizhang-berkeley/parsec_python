"""Nonlocal forces of the k-points in sectors from one atom per orbit of their little groups (``PARSEC_KPOINT_FORCE_ORBITS``).

The silicon cube with one atom moved along x (C2x and the two mirrors through x keep it): the k-point SCF at Gamma
and X in the sectors of their little groups, its nonlocal forces from the representative atoms turned onto their
orbits against those of every atom (``off``, the route of before).  The completion itself is checked on a model.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.SCF import kpoint_sectors as sectors


class CompletionTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_FORCE_ORBITS": value}):
                self.assertEqual(sectors.kpoint_force_orbits_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_FORCE_ORBITS": "on"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_FORCE_ORBITS"):
                sectors.kpoint_force_orbits_setting()

    def test_the_orbit_of_a_representative(self) -> None:
        # the mirror x -> -x: atoms 0 and 1 its orbit, atom 2 on the plane
        matrices = np.array([np.eye(3), np.diag([-1.0, 1.0, 1.0])])
        images = np.array([[0, 1, 2], [1, 0, 2]])
        forces = np.array([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0], [0.0, 0.5, -0.4]])
        out = sectors.KPointSectors.complete_orbit_forces(forces, (np.array([0, 2]), images, matrices))
        np.testing.assert_array_equal(out, [[0.3, -0.2, 0.1], [-0.3, -0.2, 0.1], [0.0, 0.5, -0.4]])
        with self.assertRaisesRegex(RuntimeError, "miss an atom"):
            sectors.KPointSectors.complete_orbit_forces(forces, (np.array([0]), images, matrices))


def setUpModule() -> None:
    # the cell is small: split it all the same
    patcher = mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class OrbitForceTests(unittest.TestCase):
    def test_the_forces_of_the_orbits_equal_those_of_every_atom(self) -> None:
        from parsec_python.Forces import nonlocal_force
        from parsec_python.SCF import kpoints as kpoint_module
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
        problem = replace(
            base,
            scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
            eigensolver=replace(base.eigensolver, tolerance=1.0e-10),
        )
        atoms = list(problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
        problem = replace(problem, atoms=atoms)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        weights = np.array([0.5, 0.5])
        built = []
        original = nonlocal_force.build_periodic_projector_gradients

        def spy(*arguments, **keywords):
            built.append(keywords.get("atom_indices"))
            return original(*arguments, **keywords)

        results = {}
        for setting in ("off", "auto"):
            built.clear()
            switches = {"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_FORCE_ORBITS": setting}
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings(), \
                    mock.patch.object(kpoint_module, "build_periodic_projector_gradients", side_effect=spy):
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
                results[setting] = run_scf_kpoints(system, k_points, weights, compute_forces=True)
            results[setting + "_built"] = list(built)
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], 2.0)
        # off builds no gradients of chosen atoms; auto those of the representatives (fewer than the 8 atoms)
        self.assertEqual(results["off_built"], [])
        self.assertTrue(results["auto_built"])
        self.assertTrue(all(chosen is not None and len(chosen) < 8 for chosen in results["auto_built"]))
        self.assertGreater(float(np.max(np.abs(off.nonlocal_forces))), 1.0e-3)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-10)
        difference = float(np.max(np.abs(auto.nonlocal_forces - off.nonlocal_forces)))
        print(f"\n nonlocal forces, orbits - every atom: {difference:.3e} Ry/bohr")
        np.testing.assert_allclose(auto.nonlocal_forces, off.nonlocal_forces, rtol=0.0, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
