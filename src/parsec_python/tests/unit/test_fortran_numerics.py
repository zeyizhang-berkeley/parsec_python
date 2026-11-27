"""``PARSEC_FORTRAN_NUMERICS``: the numerics of PARSEC 1.5 in one switch.

Checked here (CPU): the values of the umbrella (an unknown value is refused, also by the input reader); under ``on``
every switch it lists reads PARSEC 1.5's value, which is a value its own setting accepts, and its default under
``off``; a switch set in the environment wins; the Hartree boundary keywords default to PARSEC's boundary values for
an input that gives its radius, not for a radius left to the default rule, and a keyword of the input wins; parsec.out
names the umbrella in one line under ``on`` only.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parsec_python.fortran_numerics import (
    FORTRAN_NUMERICS_ENV,
    FORTRAN_NUMERICS_VALUES,
    fortran_numerics_setting,
    switch_value,
)

_DATA = Path(__file__).resolve().parents[1] / "data"


def _settings():
    """The setting function of every switch of the umbrella."""

    from parsec_python.Forces.wire_slab import wire_slab_gaussian_force_setting, wire_slab_gaussian_setting
    from parsec_python.Grid.cluster import sphere_ties_setting
    from parsec_python.Hartree.variational import hartree_variational_setting
    from parsec_python.Input.parsec_input import cubic_spline_default_setting
    from parsec_python.Hamiltonian.hubbard import ldau_smooth_projectors_fraction
    from parsec_python.Occupations.fermi_dirac import fermi_gap_root_setting
    from parsec_python.Pseudopotential.fhipp import fhi_origin_setting
    from parsec_python.Pseudopotential.local_filter import local_filter_setting
    from parsec_python.Pseudopotential.potre import projector_tail_setting, spline_coulomb_tail_setting
    from parsec_python.Pseudopotential.radial_derivatives import spline_force_center_setting
    from parsec_python.Pseudopotential.radial_quadrature import kb_quadrature_setting
    from parsec_python.SCF.filter_degree import filter_degree_stall_setting
    from parsec_python.SCF.bands import band_buffer_setting, band_convergence_setting
    from parsec_python.SCF.wire_slab import gaussian_solve_setting
    from parsec_python.V_ion.ionic_potential import projector_center_setting
    from parsec_python.V_xc.ca_lda import ca_continuity_setting
    from parsec_python.V_xc.noncollinear import noncollinear_gga_frame_setting
    from parsec_python.V_xc.nuclear_gradient import nuclear_gradient_setting

    table = {
        "PARSEC_CUBIC_SPLINE_DEFAULT": (cubic_spline_default_setting, "on"),
        "PARSEC_PROJECTOR_TAIL": (projector_tail_setting, "auto"),
        "PARSEC_KB_QUADRATURE": (kb_quadrature_setting, "auto"),
        "PARSEC_CA_CONTINUITY": (ca_continuity_setting, "on"),
        "PARSEC_HARTREE_VARIATIONAL": (hartree_variational_setting, "slab"),
        "PARSEC_PROJECTOR_CENTER": (projector_center_setting, "auto"),
        "PARSEC_SPLINE_FORCE_CENTER": (spline_force_center_setting, "auto"),
        "PARSEC_WIRE_SLAB_GAUSSIAN": (wire_slab_gaussian_setting, "auto"),
        "PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE": (gaussian_solve_setting, "auto"),
        "PARSEC_WIRE_SLAB_GAUSSIAN_FORCE": (wire_slab_gaussian_force_setting, "auto"),
        "PARSEC_BAND_CONVERGENCE": (band_convergence_setting, "auto"),
        # band_buffer_setting returns the fraction of auto, None for off
        "PARSEC_BAND_BUFFER": (band_buffer_setting, 0.6),
        "PARSEC_FERMI_GAP_ROOT": (fermi_gap_root_setting, "on"),
        "PARSEC_FHI_ORIGIN": (fhi_origin_setting, "auto"),
        "PARSEC_FILTER_DEGREE_STALL": (filter_degree_stall_setting, "off"),
        "PARSEC_SPLINE_COULOMB_TAIL": (spline_coulomb_tail_setting, "auto"),
        "PARSEC_LOCAL_FILTER": (local_filter_setting, "all"),
        # ldau_smooth_projectors_fraction returns the width of the taper, 0.0 for off
        "PARSEC_LDAU_SMOOTH_PROJECTORS": (ldau_smooth_projectors_fraction, 0.2),
        "PARSEC_SPHERE_TIES": (sphere_ties_setting, "symmetric"),
        "PARSEC_GGA_NUCLEAR_GRADIENT": (nuclear_gradient_setting, "off"),
        "PARSEC_NONCOLLINEAR_GGA_FRAME": (noncollinear_gga_frame_setting, "auto"),
    }
    return table


def _environment(**values):
    """The environment without any switch of the umbrella, with ``values``."""

    names = set(FORTRAN_NUMERICS_VALUES) | {FORTRAN_NUMERICS_ENV}
    environment = {name: value for name, value in os.environ.items() if name not in names}
    environment.update(values)
    return mock.patch.dict(os.environ, environment, clear=True)


class UmbrellaSettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with _environment():
            self.assertEqual(fortran_numerics_setting(), "off")
        for value in ("on", " ON ", "off"):
            with _environment(PARSEC_FORTRAN_NUMERICS=value):
                self.assertEqual(fortran_numerics_setting(), value.strip().lower())
        with _environment(PARSEC_FORTRAN_NUMERICS="1"):
            with self.assertRaisesRegex(ValueError, FORTRAN_NUMERICS_ENV):
                fortran_numerics_setting()
            # a switch of the umbrella that is not set asks for it
            with self.assertRaisesRegex(ValueError, FORTRAN_NUMERICS_ENV):
                switch_value("PARSEC_CA_CONTINUITY", "on")

    def test_every_switch_follows_the_umbrella(self) -> None:
        settings = _settings()
        self.assertEqual(set(settings), set(FORTRAN_NUMERICS_VALUES))
        for name, (setting, default) in settings.items():
            with self.subTest(switch=name):
                with _environment():
                    self.assertEqual(setting(), default)
                with _environment(PARSEC_FORTRAN_NUMERICS="off"):
                    self.assertEqual(setting(), default)
                with _environment(PARSEC_FORTRAN_NUMERICS="on"):
                    value = setting()
                    expected = {"PARSEC_BAND_BUFFER": None, "PARSEC_LDAU_SMOOTH_PROJECTORS": 0.0}.get(
                        name, FORTRAN_NUMERICS_VALUES[name]
                    )
                    self.assertEqual(value, expected)
                # the value of PARSEC 1.5 set alone is the same
                with _environment(**{name: FORTRAN_NUMERICS_VALUES[name]}):
                    self.assertEqual(setting(), value)

    def test_fermi_level_and_fhi_origin_follow_the_umbrella(self) -> None:
        # the two settings read their switch through switch_value: unset, the umbrella turns both off
        from parsec_python.Occupations.fermi_dirac import fermi_gap_root_setting
        from parsec_python.Pseudopotential.fhipp import fhi_origin_setting

        with _environment():
            self.assertEqual((fermi_gap_root_setting(), fhi_origin_setting()), ("on", "auto"))
        with _environment(PARSEC_FORTRAN_NUMERICS="on"):
            self.assertEqual((fermi_gap_root_setting(), fhi_origin_setting()), ("off", "off"))
        with _environment(PARSEC_FORTRAN_NUMERICS="on", PARSEC_FERMI_GAP_ROOT="on", PARSEC_FHI_ORIGIN="auto"):
            self.assertEqual((fermi_gap_root_setting(), fhi_origin_setting()), ("on", "auto"))

    def test_an_explicit_switch_wins(self) -> None:
        settings = _settings()
        for name, (setting, default) in settings.items():
            with self.subTest(switch=name):
                text = "auto" if name == "PARSEC_BAND_BUFFER" else str(default)
                with _environment(PARSEC_FORTRAN_NUMERICS="on", **{name: text}):
                    self.assertEqual(setting(), default)
        with _environment(PARSEC_FORTRAN_NUMERICS="on", PARSEC_KB_QUADRATURE="logmesh"):
            self.assertEqual(settings["PARSEC_KB_QUADRATURE"][0](), "logmesh")
        # a switch outside the umbrella is not touched by it
        with _environment(PARSEC_FORTRAN_NUMERICS="on"):
            self.assertEqual(switch_value("PARSEC_KPOINT_FOLD", "shortest"), "shortest")

    def test_parsec_out_line(self) -> None:
        from parsec_python.Output.parsec_output import fortran_numerics_lines

        with _environment():
            self.assertEqual(fortran_numerics_lines(), ())
        with _environment(PARSEC_FORTRAN_NUMERICS="on"):
            (line,) = fortran_numerics_lines()
            self.assertIn("PARSEC_FORTRAN_NUMERICS=on", line)


_H2 = (
    "Grid_Spacing: 0.5\nStates_Num: 4\nAtom_Types_Num: 1\nAtom_Type: H\nLocal_Component: s\n"
    "begin Atom_Coord\n 0.0 0.0 0.0\n 1.4 0.0 0.0\nend Atom_Coord\n"
)


class InputTests(unittest.TestCase):
    def _parse(self, text, **environment):
        from parsec_python.Input import parse_parsec_input

        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(_DATA / "water_H_POTRE.DAT", Path(folder) / "H_POTRE.DAT")
            path = Path(folder) / "parsec.in"
            path.write_text(text)
            with _environment(**environment):
                return parse_parsec_input(path).problem.hartree

    def test_unknown_value_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, FORTRAN_NUMERICS_ENV):
            self._parse("Boundary_Sphere_Radius: 6.0\n" + _H2, PARSEC_FORTRAN_NUMERICS="yes please")

    def test_hartree_boundary_keywords(self) -> None:
        given = "Boundary_Sphere_Radius: 6.0\n" + _H2
        default = self._parse(given)
        self.assertEqual((default.boundary_tolerance is not None, default.atomic_tail), (True, "auto"))
        fortran = self._parse(given, PARSEC_FORTRAN_NUMERICS="on")
        self.assertEqual((fortran.boundary_tolerance, fortran.atomic_tail), (None, "off"))
        # a keyword of the input wins
        kept = self._parse("Hartree_Boundary_Tolerance: 1e-3 Ry\nHartree_Atomic_Tail: auto\n" + given,
                           PARSEC_FORTRAN_NUMERICS="on")
        self.assertEqual((kept.boundary_tolerance, kept.atomic_tail), (default.boundary_tolerance, "auto"))
        # a radius left to the default rule keeps the boundary the rule chooses it for
        ruled = self._parse(_H2, PARSEC_FORTRAN_NUMERICS="on")
        self.assertEqual(ruled, self._parse(_H2))


if __name__ == "__main__":
    unittest.main()
