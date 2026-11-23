"""The interpolation of a species without ``Cubic_Spline`` in the input (``PARSEC_CUBIC_SPLINE_DEFAULT``).

``on`` (the default): such a species takes PARSEC's cubic spline, as with ``Cubic_Spline: .true.``; ``off``: the
linear tables of PARSEC 1.5 (``usrinputfile.F90`` reads the label with the default ``.false.``).  Checked here (CPU):

* the values of the switch, refused when the input is read;
* the flag of every species under both values, a value of the input winning over both, the positional reading of
  the label (the n-th ``Cubic_Spline`` goes to the n-th ``Atom_Type``), and ``Old_Interpolation_Format``, which keeps
  the linear tables of a cluster;
* the line of parsec.out (only where the default gave a species the spline) and the effective input of
  ``Generate_Complete_Parsec_In``, which writes the value of every species, so that it reproduces the run under
  either value of the switch;
* the routines of a bulk cell with forces at the Gamma point: those of before under both values (Andrew Grieder's
  for a species without the label, the Fortran-derived ones for ``Cubic_Spline: .true.``).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parsec_python.Input.parsec_input import (
    CUBIC_SPLINE_DEFAULT_ENV,
    ParsecInputError,
    cubic_spline_default_setting,
    parse_parsec_input,
)
from parsec_python.Input.routines import GAMMA_FORCES_FEATURE
from parsec_python.Output.parsec_output import cubic_spline_default_lines

_DATA = Path(__file__).resolve().parents[1] / "data"

_CLUSTER = """Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.5
Correlation_Type: ca
States_Num: 4
{extra}
Atom_Types_Num: 2
Atom_Type: H
Local_Component: s
{first}
begin Atom_Coord
0.0 0.0 0.7
end Atom_Coord
Atom_Type: Hb
Element_Symbol: H
Local_Component: s
{second}
begin Atom_Coord
0.0 0.0 -0.7
end Atom_Coord
"""

_BULK = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0  6.0  6.0
end Cell_Shape
Grid_Spacing: 0.5
Expansion_Order: 6
Ignore_Symmetry: .true.
Correlation_Type: ca
States_Num: 4
Skip_Force: .false.
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  2.2  3.0  3.1
end Atom_Coord
"""


def _setting(value):
    return mock.patch.dict(os.environ, {CUBIC_SPLINE_DEFAULT_ENV: value})


class CubicSplineDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        for name in ("H_POTRE.DAT", "Hb_POTRE.DAT"):
            (self.directory / name).write_bytes((_DATA / "H_POTRE.DAT").read_bytes())

    def parse(self, text: str, name: str = "parsec.in"):
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)

    def cluster(self, first: str = "", second: str = "", extra: str = ""):
        return self.parse(_CLUSTER.format(first=first, second=second, extra=extra))

    @staticmethod
    def flags(translation):
        return {symbol: species.use_spline for symbol, species in translation.problem.pseudopotentials.items()}

    def test_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(CUBIC_SPLINE_DEFAULT_ENV, None)
            self.assertEqual(cubic_spline_default_setting(), "on")
        for value in ("on", "off", " OFF ", ""):
            with _setting(value):
                self.assertEqual(cubic_spline_default_setting(), value.strip().lower() or "on")
        with _setting("linear"):
            with self.assertRaisesRegex(ValueError, CUBIC_SPLINE_DEFAULT_ENV):
                cubic_spline_default_setting()
            # refused when the input is read, also where every species gives the label
            with self.assertRaisesRegex((ValueError, ParsecInputError), CUBIC_SPLINE_DEFAULT_ENV):
                self.cluster("Cubic_Spline: .true.", "Cubic_Spline: .true.")

    def test_species_without_the_label_take_the_default(self):
        with _setting("on"):
            translation = self.cluster()
            self.assertEqual(self.flags(translation), {"H": True, "Hb": True})
            self.assertEqual(translation.cubic_spline_defaulted, ("H", "Hb"))
        with _setting("off"):
            translation = self.cluster()
            self.assertEqual(self.flags(translation), {"H": False, "Hb": False})
            self.assertEqual(translation.cubic_spline_defaulted, ("H", "Hb"))

    def test_a_value_of_the_input_wins(self):
        for value in ("on", "off"):
            with self.subTest(value), _setting(value):
                translation = self.cluster("Cubic_Spline: .false.", "Cubic_Spline: .true.")
                self.assertEqual(self.flags(translation), {"H": False, "Hb": True})
                self.assertEqual(translation.cubic_spline_defaulted, ())
                # the n-th label goes to the n-th species: the second species has none
                translation = self.cluster("Cubic_Spline: .false.")
                self.assertEqual(self.flags(translation), {"H": False, "Hb": value == "on"})
                self.assertEqual(translation.cubic_spline_defaulted, ("Hb",))

    def test_old_interpolation_format_keeps_the_linear_tables_of_a_cluster(self):
        with _setting("on"):
            translation = self.cluster(extra="Old_Interpolation_Format: .true.")
            self.assertEqual(self.flags(translation), {"H": False, "Hb": False})
            self.assertEqual(cubic_spline_default_lines(translation),
                             (" Linear tables for H Hb (Old_Interpolation_Format)",))
            # a value of the input still wins
            translation = self.cluster("Cubic_Spline: .true.", extra="Old_Interpolation_Format: .true.")
            self.assertEqual(self.flags(translation), {"H": True, "Hb": False})

    def test_the_line_of_parsec_out(self):
        with _setting("on"):
            self.assertEqual(
                cubic_spline_default_lines(self.cluster("Cubic_Spline: .false.")),
                (" Cubic spline (default) for Hb; PARSEC_CUBIC_SPLINE_DEFAULT=off: linear",),
            )
            self.assertEqual(cubic_spline_default_lines(self.cluster("Cubic_Spline: .true.", "Cubic_Spline: .true.")),
                             ())
        with _setting("off"):
            # PARSEC 1.5's default: the output of before
            self.assertEqual(cubic_spline_default_lines(self.cluster()), ())

    def test_the_complete_input_reproduces_the_run_under_either_value(self):
        for value in ("on", "off"):
            with self.subTest(value):
                with _setting(value):
                    translation = self.cluster("Cubic_Spline: .false.", extra="Generate_Complete_Parsec_In: .true.")
                text = translation.complete_input
                self.assertIsNotNone(text)
                for other in ("on", "off"):
                    with _setting(other):
                        again = self.parse(text, f"complete_{value}_{other}.in")
                    self.assertEqual(self.flags(again), self.flags(translation))
                    self.assertEqual(again.cubic_spline_defaulted, ())
        # a bulk cell with forces at the Gamma point: the spline by default kept Andrew Grieder's routines, which the
        # written input keeps (its Cubic_Spline .true. alone would take the Fortran-derived ones)
        for value in ("on", "off"):
            with self.subTest(bulk=value):
                with _setting(value):
                    translation = self.parse(_BULK.replace("Skip_Force", "Generate_Complete_Parsec_In: .true.\nSkip_Force"))
                self.assertFalse(translation.routines.fortran)
                for other in ("on", "off"):
                    with _setting(other):
                        again = self.parse(translation.complete_input, f"complete_bulk_{value}_{other}.in")
                    self.assertEqual(self.flags(again), self.flags(translation))
                    self.assertEqual(again.routines.fortran, translation.routines.fortran)
                    self.assertEqual(again.problem.grid.fortran_routines, translation.problem.grid.fortran_routines)

    def test_gamma_point_forces_of_a_bulk_cell_keep_their_routines(self):
        # Cubic_Spline: .true. in the input takes the Fortran-derived routines (the default of Gamma-point forces of
        # Cubic_Spline species); the spline by default keeps Andrew Grieder's under both values, as before
        for value in ("on", "off"):
            with self.subTest(value), _setting(value):
                translation = self.parse(_BULK)
                self.assertFalse(translation.routines.fortran)
                self.assertEqual(self.flags(translation), {"H": value == "on"})
                explicit = self.parse(
                    _BULK.replace("Local_Component: s\n", "Local_Component: s\nCubic_Spline: .true.\n")
                )
                self.assertTrue(explicit.routines.fortran)
                self.assertEqual(explicit.routines.defaults, (GAMMA_FORCES_FEATURE,))


if __name__ == "__main__":
    unittest.main()
