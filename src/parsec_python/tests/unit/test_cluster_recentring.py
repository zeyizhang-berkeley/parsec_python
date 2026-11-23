"""The report of a re-centred cluster (``initial.f90:163-182``, ``Output/parsec_output.py``).

PARSEC translates a cluster whose bounding box has its midpoint more than 0.1 bohr from the origin, warns and prints
the new coordinates, as the reference outputs of ``examples/0d_AuH`` show; the forces it reports are those of the
moved atoms.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.Output.parsec_output import cluster_recentring_lines

DATA = Path(__file__).resolve().parents[1] / "data"
THREE = ((-2.131, -0.913, 0.217), (2.452, -1.187, -0.163), (0.209, 2.996, 0.081))


def _translation(positions, extra=""):
    folder = Path(tempfile.mkdtemp(prefix="parsec_recentre_"))
    try:
        shutil.copyfile(DATA / "water_H_POTRE.DAT", folder / "H_POTRE.DAT")
        rows = "".join(f" {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in positions)
        (folder / "parsec.in").write_text(
            "Boundary_Sphere_Radius: 8.0\nGrid_Spacing: 0.5\nStates_Num: 4\nCoordinate_Unit: Cartesian_Bohr\n"
            + extra
            + f"Atom_Types_Num: 1\nAtom_Type: H\nLocal_Component: s\nbegin Atom_Coord\n{rows}end Atom_Coord\n"
        )
        return parse_parsec_input(folder / "parsec.in")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class ClusterRecentringReportTests(unittest.TestCase):
    def test_a_moved_cluster_prints_the_new_coordinates(self) -> None:
        lines = cluster_recentring_lines(_translation(THREE))
        positions = np.array(THREE)
        shift = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
        self.assertEqual(
            lines[:5],
            ["", " WARNING: atom coordinates were shifted so", " that the cluster is centered at the origin", "",
             " The new atom coordinates are:"],
        )
        moved = np.array([[float(v) for v in line.split()] for line in lines[5:]])
        np.testing.assert_allclose(moved, positions - shift, atol=1e-9)
        self.assertEqual(lines[5], "   -2.291500000      -1.817500000       0.190000000")

    def test_the_block_of_the_fortran_reference_of_auh(self) -> None:
        # examples/0d_AuH: the lines of the block in fortran_reference.out (PARSEC 1.5), the port's from its inputs
        root = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "reduced_perturbative_soc"
        if not (root / "fortran_reference.out").exists():
            self.skipTest("examples/0d_AuH is not in this tree")
        text = (root / "fortran_reference.out").read_text(errors="replace").splitlines()
        start = text.index(" WARNING: atom coordinates were shifted so")
        reference = text[start:start + 6]
        for name in ("parsec.in", "parsec_fortran.in"):
            lines = cluster_recentring_lines(parse_parsec_input(root / name))
            self.assertEqual(lines[1:], reference, name)

    def test_nothing_where_the_atoms_stay(self) -> None:
        positions = np.array(THREE)
        centred = positions - 0.5 * (positions.min(axis=0) + positions.max(axis=0)) + 0.03
        self.assertEqual(cluster_recentring_lines(_translation(centred)), [])
        self.assertEqual(
            cluster_recentring_lines(_translation(THREE, "Recenter_Atoms_In_Aperiodic_Directions: .false.\n")), []
        )


if __name__ == "__main__":
    unittest.main()
