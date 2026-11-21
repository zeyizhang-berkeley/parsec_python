"""Charged sheets and the space group of a slab from PARSEC's own search (band structure, spin-orbit, NSCF; 2026-11-21).

A slab with a band structure, spin-orbit coupling or an NSCF run takes the group of PARSEC's wire and slab search
(``symmetries.f90``, :func:`parsec_python.Grid.space_group.find_wire_slab_space_group`), which looks at the atoms
alone.  Charged sheets (``add_charged_sheets``) add the potential ``sum_s 4 pi (q_s/A) |z - z_s|``, so a capacitor whose
sheets are not their own mirror image breaks ``z -> -z``: up to 2026-11-20 that operation still reduced the mesh and
averaged the density, which erased the dipole of the capacitor.  The operations that do not map the sheets onto
sheets of the same charge are dropped now (:func:`parsec_python.Grid.space_group.sheet_preserving_operations`).
Checked here (CPU):

* :func:`sheet_preserving_operations` on mirrors about ``z = 0`` and about another plane;
* the parser: a square carbon slab with ``bandstruc`` keeps D4h without sheets and with symmetric ones, and only the
  eight operations that keep ``z`` with an asymmetric capacitor (with a warning);
* the k-point SCF of the reference solver: the capacitor on the wedge of the remaining group has the energy and the
  dipole of the whole mesh without symmetry on the same grid.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling
from parsec_python.Grid.space_group import SpaceGroup, sheet_preserving_operations
from parsec_python.tests.unit.test_slab_mesh_symmetry import _Directory, _sheets

_SLAB = """\
Periodic_System: .true.
Boundary_Conditions: slab
begin Cell_Shape
  5.0 0.0 0.0
  0.0 5.0 0.0
end Cell_Shape
Boundary_Sphere_Radius: 5.00
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.45
Expansion_Order: 6
Correlation_Type: ca
States_Num: 6
Fermi_Temp: 500 K
Convergence_Criterion: 1e-8 Ry
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
3 3 1
end Monkhorst_Pack_Grid
{bands}
{extra}
Atom_Types_Num: 1

Atom_Type: C
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 1.6
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
begin Atom_Coord
   0.0 0.0 0.0
end Atom_Coord
"""

_BANDS = "begin bandstruc\n2 0 0 0 0.5 0 0 G-X\nend bandstruc"
_CAPACITOR = ((0.05, -4.0), (-0.05, 4.0))


def _mirror_group(translation_z: float = 0.0) -> SpaceGroup:
    """The identity and ``z -> -z`` (with the fractional translation ``translation_z`` of ``alatt`` along ``z``)."""

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    mirror = ((1, 0, 0), (0, 1, 0), (0, 0, -1))
    return SpaceGroup(
        rotations=(identity, mirror),
        translations=((0.0, 0.0, 0.0), (0.0, 0.0, translation_z)),
        cartesian_rotations=tuple(tuple(tuple(float(v) for v in row) for row in m) for m in (identity, mirror)),
        labels=("E  ", "IC2"),
    )


class SheetOperationTests(unittest.TestCase):
    def test_mirrors(self) -> None:
        alatt = np.diag([5.0, 5.0, 20.0])
        group = _mirror_group()
        for sheets, order in (
            (((0.05, -4.0), (-0.05, 4.0)), 1),  # a capacitor: the mirror exchanges opposite charges
            (((0.05, -4.0), (0.05, 4.0)), 2),
            (((0.1, 0.0),), 2),
            (((0.1, 1.0),), 1),
            ((), 2),
        ):
            with self.subTest(sheets=sheets):
                kept, dropped = sheet_preserving_operations(group, sheets, alatt)
                self.assertEqual((kept.order, dropped), (order, 2 - order))
                if dropped:
                    self.assertFalse(kept.full_holohedry)
        # a mirror about z = 2 (r -> U (r - A t), A t = (0, 0, 4)) maps a sheet at z = 1 onto the one at z = 3
        shifted = _mirror_group(4.0 / 20.0)
        self.assertEqual(sheet_preserving_operations(shifted, ((0.1, 1.0), (0.1, 3.0)), alatt)[0].order, 2)
        self.assertEqual(sheet_preserving_operations(shifted, ((0.1, 1.0), (0.1, -1.0)), alatt)[0].order, 1)


class ParserTests(_Directory):
    def slab(self, extra=""):
        return self.parse_text(_SLAB.format(bands=_BANDS, extra=extra))

    def test_the_sheets_decide_the_mirror(self) -> None:
        plain = self.slab()
        self.assertEqual(plain.problem.space_group.order, 16)
        self.assertTrue(np.any(plain.problem.space_group.rotation_matrices[:, 2, 2] == -1))
        symmetric = self.slab(_sheets((0.05, -4.0), (0.05, 4.0)))
        self.assertEqual(symmetric.problem.space_group.order, 16)
        capacitor = self.slab(_sheets(*_CAPACITOR))
        group = capacitor.problem.space_group
        self.assertEqual(group.order, 8)
        self.assertTrue(np.all(group.rotation_matrices[:, 2, 2] == 1))
        self.assertTrue(any("charged sheets" in note and "8 symmetry operation" in note for note in capacitor.warnings))
        self.assertFalse(any("charged sheets (add_charged_sheets)" in note for note in symmetric.warnings))


class CapacitorScfTests(_Directory):
    def test_the_wedge_keeps_the_energy_and_the_dipole(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        results = {}
        # the whole mesh on the same grid: without a band structure PARSEC 1.5 searches no group for a slab (and
        # neither does PARSEC_SLAB_MESH_SYMMETRY=off); Ignore_Symmetry would also take the unshifted grid
        for label, bands, setting in (("group", _BANDS, None), ("none", "", "off")):
            problem = self.parse_text(_SLAB.format(bands=bands, extra=_sheets(*_CAPACITOR)), setting).problem
            k_points, weights = kpoint_sampling(problem)
            system = prepare_wire_slab_single_point(problem)
            result = run_scf_kpoints(system, k_points, weights)
            self.assertTrue(result.converged, label)
            z = np.asarray(system.grid.coordinates)[:, 2]
            dipole = float(np.sum(np.asarray(result.density) * z) * system.grid.volume_element)
            results[label] = (problem.space_group, len(k_points), float(result.energies.total), dipole)
        group, count, energy, dipole = results["group"]
        self.assertEqual(group.order, 8)
        self.assertLess(count, results["none"][1])
        self.assertGreater(abs(results["none"][3]), 0.05)
        # the operations left are signed permutations of the square: the wedge is the whole mesh to round-off (A100:
        # 4e-11 Ry and 2e-10 e bohr; the 16 operations of D4h had given 6.7e-3 Ry lower and no dipole)
        self.assertAlmostEqual(energy, results["none"][2], delta=2.0e-7)
        self.assertAlmostEqual(dipole, results["none"][3], delta=1.0e-5)


if __name__ == "__main__":
    unittest.main()
