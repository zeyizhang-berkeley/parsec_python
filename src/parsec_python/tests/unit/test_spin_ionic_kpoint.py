"""PARSEC_SPIN_IONIC_KPOINT: the ionic loops of a spin-polarized bulk cell at the Gamma point on the Fortran-derived
routines take the 1x1x1 k-point run of the device k-point solver (``auto``, the default since 2026-11-21) or the
Gamma-point loop of the host (``off``, as before); which inputs take it (``parsec_python.cli.spin_ionic_kpoint_sampling``)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import cli
from parsec_python.Input.parsec_input import parse_parsec_input

_DATA = Path(__file__).resolve().parents[1] / "data"

_CELL = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0  6.0  6.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.5
Expansion_Order: 6
Correlation_Type: {functional}
Spin_Polarization: .true.
States_Num: 4
Convergence_Criterion: 1.0e-6
Skip_Force: .false.
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
Initial_Spin_Polarization: 0.5
begin Atom_Coord
  2.2  3.0  3.1
  3.9  3.1  2.9
end Atom_Coord
"""


class SpinIonicKPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        (self.directory / "H_POTRE.DAT").write_bytes((_DATA / "spin_pbe_H_POTRE.DAT").read_bytes())

    def parse(self, extra: str = "", *, functional: str = "pbe"):
        path = self.directory / "parsec.in"
        path.write_text(_CELL.format(extra=extra, functional=functional), encoding="utf-8")
        return parse_parsec_input(path)

    def sampling(self, translation, *, device=True, restart=None, k_sampling=None):
        with mock.patch.object(cli, "_gamma_spin_on_kpoint_driver", return_value=device):
            return cli.spin_ionic_kpoint_sampling(translation, k_sampling, True, False, restart=restart)

    def test_spin_pbe_forces_take_the_fortran_routines_and_the_mesh_of_gamma(self) -> None:
        translation = self.parse()
        self.assertTrue(translation.routines.fortran)  # the default of the forces of a spin PBE cell
        k_points, weights = self.sampling(translation)
        np.testing.assert_array_equal(k_points, np.zeros((1, 3)))
        np.testing.assert_array_equal(weights, np.ones(1))

    def test_off_and_the_inputs_it_leaves_alone(self) -> None:
        translation = self.parse()
        with mock.patch.dict(os.environ, {cli.SPIN_IONIC_KPOINT_ENV: "off"}):
            self.assertIsNone(self.sampling(translation))
        self.assertIsNone(self.sampling(translation, device=False))  # no device k-point backend
        # defensive: the parser refuses a self-consistent Restart_Run of a periodic spin cell before this is asked
        self.assertIsNone(self.sampling(translation, restart=object()))
        given = (np.full((2, 3), 0.25), np.full(2, 0.5))
        self.assertIs(self.sampling(translation, k_sampling=given), given)

    def test_the_excluded_inputs_parse_and_keep_the_gamma_loop(self) -> None:
        from dataclasses import replace

        (self.directory / "occup.in").write_text(
            "spin up\n  1\n  0\nspin down\n  1\n  0\n", encoding="ascii"
        )
        # TS and Symmetrize_Charge_Density need Ignore_Symmetry on this cell (PARSEC error 159; the operations of
        # its two atoms do not map the grid): the control input with Ignore_Symmetry alone takes the k-point run
        control = self.parse("Ignore_Symmetry: .true.")
        self.assertIsNotNone(self.sampling(control))
        parsed = {
            "TS": self.parse("Ignore_Symmetry: .true.\nVan_Der_Waals: .true."),
            "Symmetrize_Charge_Density": self.parse("Ignore_Symmetry: .true.\nSymmetrize_Charge_Density: .true."),
            "meta-GGA": self.parse(functional="scan"),
            "hybrid": self.parse(functional="pbe0"),
            "occup.in": self.parse("Fermi_Temp: -1"),
        }
        flags = {
            "TS": lambda scf: scf.van_der_waals,
            "Symmetrize_Charge_Density": lambda scf: scf.symmetrize_density,
            "meta-GGA": lambda scf: scf.is_meta_gga,
            "hybrid": lambda scf: scf.is_hybrid,
            "occup.in": lambda scf: scf.fixed_occupations is not None,
        }
        for name, other in parsed.items():
            with self.subTest(excluded=name):
                self.assertTrue(flags[name](other.problem.scf))  # the input holds what is excluded
                self.assertTrue(other.problem.scf.spin_polarized)
                self.assertIsNone(self.sampling(other))
        # what the parser cannot give this cell of H atoms: LDA+U (a d channel), spin-orbit species and spinors,
        # set on the parsed default input
        translation = self.parse()
        problem = translation.problem
        species = dict(problem.pseudopotentials)
        species["H"] = replace(species["H"], hubbard_u=0.3)
        changed = {
            "LDA+U": replace(translation, problem=replace(problem, pseudopotentials=species)),
            "spin-orbit": replace(
                translation, problem=replace(problem, scf=replace(problem.scf, self_consistent_spin_orbit=True))
            ),
            "non-collinear": replace(
                translation, problem=replace(problem, scf=replace(problem.scf, noncollinear=True))
            ),
            "unpolarized": replace(
                translation, problem=replace(problem, scf=replace(problem.scf, spin_polarized=False))
            ),
            # defensive: the parser refuses Energy_Refinement of a periodic input
            "Energy_Refinement": replace(translation, energy_refinement="b3lyp"),
        }
        self.assertIsNotNone(self.sampling(translation))
        for name, other in changed.items():
            with self.subTest(excluded=name):
                self.assertIsNone(self.sampling(other))

    def test_the_setting_reads_its_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {cli.SPIN_IONIC_KPOINT_ENV: value}):
                self.assertEqual(cli.spin_ionic_kpoint_setting(), expected)
        with mock.patch.dict(os.environ, {cli.SPIN_IONIC_KPOINT_ENV: "bogus"}), self.assertRaisesRegex(
            ValueError, cli.SPIN_IONIC_KPOINT_ENV
        ):
            cli.spin_ionic_kpoint_setting()


if __name__ == "__main__":
    unittest.main()
