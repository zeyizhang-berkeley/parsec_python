"""The parse note of two ``Atom_Type`` labels of one species whose labels hide operations of the atoms from the
symmetry sectors of a spin-polarized run (``parsec_input._identical_label_note``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parsec_python.Input.parsec_input import parse_parsec_input

_DATA = Path(__file__).resolve().parents[1] / "data"

_CLUSTER = """Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.5
States_Num: 4
Correlation_Type: pbe
Spin_Polarization: {spin}
{extra}
"""

_CELL = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  8.0  8.0  8.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.5
States_Num: 4
Correlation_Type: pbe
Spin_Polarization: {spin}
{extra}
"""


def _type(label: str, polarization: float, z: float) -> str:
    return (
        f"Atom_Type: {label}\nElement_Symbol: H\nLocal_Component: s\nInitial_Spin_Polarization: {polarization}\n"
        f"begin Atom_Coord\n  0.0 0.0 {z}\nend Atom_Coord\n"
    )


class IdenticalLabelNoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        content = (_DATA / "spin_pbe_H_POTRE.DAT").read_bytes()
        for label in ("H", "H_a", "H_b"):
            (self.directory / f"{label}_POTRE.DAT").write_bytes(content)

    def notes(self, *, cell=False, spin=".true.", extra="", types=(("H_a", 0.1), ("H_b", 0.1))):
        centre = 4.0 if cell else 0.0
        if len(types) == 1:
            label, polarization = types[0]
            body = (
                f"Atom_Types_Num: 1\nAtom_Type: {label}\nElement_Symbol: H\nLocal_Component: s\n"
                f"Initial_Spin_Polarization: {polarization}\nbegin Atom_Coord\n  0.0 0.0 {centre + 0.7}\n"
                f"  0.0 0.0 {centre - 0.7}\nend Atom_Coord\n"
            )
        else:
            body = "Atom_Types_Num: 2\n" + "".join(
                _type(label, polarization, centre + offset)
                for (label, polarization), offset in zip(types, (0.7, -0.7))
            )
        header = (_CELL if cell else _CLUSTER).format(spin=spin, extra=extra)
        path = self.directory / "parsec.in"
        path.write_text(header + body, encoding="utf-8")
        translation = parse_parsec_input(path)
        return [text for text in translation.warnings if "are one species" in text]

    def test_two_labels_of_one_species_hide_the_mirror(self) -> None:
        for cell in (False, True):
            with self.subTest(cell=cell):
                notes = self.notes(cell=cell)
                self.assertEqual(len(notes), 1)
                self.assertIn("H_a, H_b", notes[0])
                self.assertIn("keep 8 of the 16", notes[0])  # C4v of the labeled pair, D4h of the atoms

    def test_no_note_where_the_labels_differ_or_nothing_is_hidden(self) -> None:
        for name, keywords in (
            ("opposite starts", dict(types=(("H_a", 0.5), ("H_b", -0.5)))),
            ("one label", dict(types=(("H", 0.1),))),
            ("unpolarized", dict(spin=".false.")),
            ("Ignore_Symmetry", dict(extra="Ignore_Symmetry: .true.")),
        ):
            for cell in (False, True):
                with self.subTest(case=name, cell=cell):
                    self.assertEqual(self.notes(cell=cell, **keywords), [])
        # another file content: another species
        (self.directory / "H_b_POTRE.DAT").write_bytes((self.directory / "H_a_POTRE.DAT").read_bytes() + b"\n")
        self.assertEqual(self.notes(), [])


if __name__ == "__main__":
    unittest.main()
