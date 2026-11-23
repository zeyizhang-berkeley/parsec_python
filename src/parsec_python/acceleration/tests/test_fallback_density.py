"""The density group of the full grid a spin cluster leaves its sectors for (``SCF/fallback_density``, CPU).

``PARSEC_SPIN_FALLBACK_DENSITY=abelian`` gives that full grid the abelian group of the atoms (the operations that only
change the signs of coordinates); checked here: the values of the switch, the group of an H2 pair along z (D2h, 8),
of the same pair with two labels (C2v, 4), none for a cell or a system that symmetrizes its density already.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python.acceleration.SCF.fallback_density import (
    SWITCH,
    abelian_density_group,
    fallback_density_setting,
    with_fallback_density,
)

_PAIR = """Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.5
States_Num: 4
Correlation_Type: ca
Spin_Polarization: .true.
Atom_Types_Num: {types}
{atoms}
"""
_ONE_TYPE = """Atom_Type: H
Local_Component: s
begin Atom_Coord
 0.0 0.0 0.7
 0.0 0.0 -0.7
end Atom_Coord"""
_TWO_TYPES = """Atom_Type: H_a
Element_Symbol: H
Local_Component: s
Initial_Spin_Polarization: 0.5
begin Atom_Coord
 0.0 0.0 0.7
end Atom_Coord
Atom_Type: H_b
Element_Symbol: H
Local_Component: s
Initial_Spin_Polarization: -0.5
begin Atom_Coord
 0.0 0.0 -0.7
end Atom_Coord"""
_ROOT = Path(__file__).resolve().parents[4]


def _system(atoms_block: str, types: int):
    from parsec_python.Grid import build_cluster_grid
    from parsec_python.Input.parsec_input import parse_parsec_input
    from parsec_python.V_ion import center_cluster_geometry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
        folder = Path(name)
        (folder / "parsec.in").write_text(_PAIR.format(types=types, atoms=atoms_block))
        source = (_ROOT / "src/pp_generation/NCPPs/PBE/H/PARSEC/H_POTRE.DAT").read_bytes()
        for label in ("H", "H_a", "H_b"):
            (folder / f"{label}_POTRE.DAT").write_bytes(source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(folder / "parsec.in").problem
    return SimpleNamespace(
        input=problem,
        grid=build_cluster_grid(problem.grid),
        atoms=center_cluster_geometry(problem.atoms),
        density_symmetry=None,
    )


class FallbackDensityTests(unittest.TestCase):
    def test_setting(self) -> None:
        with mock.patch.dict(os.environ, {SWITCH: ""}):
            self.assertEqual(fallback_density_setting(), "off")
        with mock.patch.dict(os.environ, {SWITCH: "Abelian"}):
            self.assertEqual(fallback_density_setting(), "abelian")
        with mock.patch.dict(os.environ, {SWITCH: "full"}):
            with self.assertRaises(ValueError):
                fallback_density_setting()

    def test_groups(self) -> None:
        plain = _system(_ONE_TYPE, 1)
        self.assertEqual(abelian_density_group(plain).order, 8)
        labelled = _system(_TWO_TYPES, 2)
        self.assertEqual(abelian_density_group(labelled).order, 4)
        cell = SimpleNamespace(input=SimpleNamespace(periodic_cell=object()), grid=None, atoms=())
        self.assertIsNone(abelian_density_group(cell))

    def test_the_system_takes_the_group_only_when_asked(self) -> None:
        system = _system(_TWO_TYPES, 2)
        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            self.assertIsNone(with_fallback_density(system))
            self.assertIsNone(system.density_symmetry)
        with mock.patch.dict(os.environ, {SWITCH: "abelian"}):
            note = with_fallback_density(system)
            self.assertIn("4 operations", note)
            self.assertEqual(system.density_symmetry.order, 4)
            # a system with a group of its own keeps it
            kept = system.density_symmetry
            self.assertIsNone(with_fallback_density(system))
            self.assertIs(system.density_symmetry, kept)

    def test_an_accelerated_system_reads_the_group_of_its_reference(self) -> None:
        from dataclasses import dataclass

        plain = _system(_TWO_TYPES, 2)

        @dataclass(frozen=True)
        class Reference:
            input: object
            grid: object
            atoms: tuple
            density_symmetry: object = None

        class Accelerated:
            def __init__(self, reference) -> None:
                self.reference = reference

            def __getattr__(self, name):
                return getattr(self.reference, name)

        system = Accelerated(Reference(plain.input, plain.grid, tuple(plain.atoms)))
        with mock.patch.dict(os.environ, {SWITCH: "abelian"}):
            self.assertIsNotNone(with_fallback_density(system))
        self.assertEqual(system.reference.density_symmetry.order, 4)
        self.assertEqual(system.density_symmetry.order, 4)

    def test_the_group_follows_the_moved_atoms(self) -> None:
        # an ionic step returns a system built from the reference the preparation keeps (without the group)
        from dataclasses import dataclass, replace

        plain = _system(_TWO_TYPES, 2)
        original = tuple(plain.atoms)

        @dataclass(frozen=True)
        class Reference:
            input: object
            grid: object
            atoms: tuple
            density_symmetry: object = None

        class Accelerated:
            def __init__(self, reference, updater=None) -> None:
                self.reference = reference
                self.geometry_updater = updater

            def __getattr__(self, name):
                return getattr(self.reference, name)

        kept = Reference(plain.input, plain.grid, original)

        def update(current, atoms):
            return Accelerated(replace(kept, atoms=tuple(atoms)), current.geometry_updater)

        system = Accelerated(kept, update)
        with mock.patch.dict(os.environ, {SWITCH: "abelian"}):
            self.assertIsNotNone(with_fallback_density(system))
            # along the axis: the group stays
            moved = system.geometry_updater(system, [replace(atom, position=np.asarray(atom.position) * 1.01)
                                                     for atom in original])
            self.assertEqual(moved.density_symmetry.order, 4)
            # off the axis: only the reflection through the plane of the pair and the axis stays (order 2)
            shifted = [replace(original[0], position=np.asarray(original[0].position) + np.array([0.1, 0.0, 0.0])),
                       *original[1:]]
            again = moved.geometry_updater(moved, shifted)
            self.assertEqual(again.density_symmetry.order, 2)


if __name__ == "__main__":
    unittest.main()
