"""The forces of a ``Kpoint_List`` on the default routines are averaged over the operations that keep the grid and the
list (``PARSEC_KPOINT_LIST_FORCE_GROUP``, since 2026-11-22), as those of the same points given as a mesh are.

Checked here (CPU, the input reader): the operations kept by :func:`parsec_python.Grid.kpoints.list_preserving_operations`
for whole meshes in either writing, time-reversal partners and reordered lists, a list that lost a point, and the
``force_space_group`` the reader gives a moved hcp Au2 cell (the 6 operations of its mesh), with ``off``,
``Ignore_Symmetry`` and without forces.  The forces themselves equal those of the mesh within 1e-9 Ry/bohr on A100
(hcp Au2, hexagonal Si4, fcc Au2; 4.2e-5 in plane for hcp Au2 without the average); the energies are those of the list.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import list_preserving_operations
from parsec_python.Grid.space_group import find_space_group
from parsec_python.Input import parse_parsec_input
from parsec_python.models import Atom

_ROOT = Path(__file__).resolve().parents[4]
_AU = _ROOT / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_A, _C = 5.45, 8.9
_CELL = np.array([[_A, 0.0, 0.0], [-_A / 2, _A * np.sqrt(3) / 2, 0.0], [0.0, 0.0, _C]])
_ATOMS = np.array([[0.0, 3.1465589671, 2.225], [2.725, 1.5732794835, 6.925]])


def _mesh(dimensions=(4, 4, 2), centred=False):
    points = []
    for index in itertools.product(*(range(n) for n in dimensions)):
        f = np.asarray(index, dtype=float) / np.asarray(dimensions, dtype=float)
        if centred:
            f = f - np.floor(f + 0.5 - 1e-12)
        points.append(f)
    return np.asarray(points)


def _input(points, *, extra="", forces=True):
    rows = "".join(f"  {p[0]:.16f} {p[1]:.16f} {p[2]:.16f}\n" for p in points)
    return (
        "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n"
        + "".join(f"{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n" for r in _CELL)
        + "end Cell_Shape\nLattice_Vector_Scale: 1.0\nGrid_Spacing: 0.3\nExpansion_Order: 6\nStates_Num: 20\n"
        + "Use_Fortran_Routines: .false.\n"
        + ("Skip_Force: .false.\n" if forces else "Skip_Force: .true.\n")
        + extra
        + "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n" + rows + "end Kpoint_List\n"
        + "Atom_Types_Num: 1\nAtom_Type: Au\nLocal_Component: s\nbegin Atom_Coord\n"
        + "".join(f"  {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n" for p in _ATOMS)
        + "end Atom_Coord\nCorrelation_Type: CA\n"
    )


def _force_group(text):
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "parsec.in").write_text(text)
        Path(folder, "Au_POTRE.DAT").write_bytes(_AU.read_bytes())
        group = parse_parsec_input(Path(folder, "parsec.in")).problem.force_space_group
    return None if group is None else group.order


class ListOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.group = find_space_group(_CELL, [Atom("Au", tuple(p)) for p in _ATOMS])
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(_CELL).T

    def _kept(self, fractions, weights=None):
        weights = np.full(len(fractions), 1.0 / len(fractions)) if weights is None else weights
        return list_preserving_operations(self.group, np.asarray(fractions) @ self.reciprocal, weights, _CELL).order

    def test_whole_meshes_keep_the_group(self) -> None:
        for centred in (False, True):
            self.assertEqual(self._kept(_mesh(centred=centred)), self.group.order)
        # reordered, and with points replaced by their time-reversal partners modulo the reciprocal lattice
        mesh = _mesh()
        rng = np.random.default_rng(3)
        changed = mesh[rng.permutation(len(mesh))]
        changed[::3] = -changed[::3] + np.array([1.0, 0.0, -2.0])
        self.assertEqual(self._kept(changed), self.group.order)

    def test_a_list_that_lost_a_point_keeps_fewer(self) -> None:
        mesh = _mesh()
        kept = self._kept(mesh[np.arange(len(mesh)) != 5])
        self.assertLess(kept, self.group.order)
        # unequal weights of a star break the group too
        weights = np.full(len(mesh), 1.0)
        weights[5] = 2.0
        self.assertLess(self._kept(mesh, weights / weights.sum()), self.group.order)

    def test_the_reader_gives_the_forces_the_group_of_the_list(self) -> None:
        self.assertEqual(_force_group(_input(_mesh())), 6)
        self.assertEqual(_force_group(_input(_mesh(centred=True))), 6)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_LIST_FORCE_GROUP": "off"}):
            self.assertIsNone(_force_group(_input(_mesh())))
        self.assertIsNone(_force_group(_input(_mesh(), extra="Ignore_Symmetry: .true.\n")))
        self.assertIsNone(_force_group(_input(_mesh(), forces=False)))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_LIST_FORCE_GROUP": "yes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_LIST_FORCE_GROUP must be auto or off"):
                _force_group(_input(_mesh()))


if __name__ == "__main__":
    unittest.main()
