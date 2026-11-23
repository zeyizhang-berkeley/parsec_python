"""The ``Cubic_Spline`` core densities of a periodic preparation (:mod:`parsec_python.acceleration.V_ion.spline_setup`)
are those of the readable loop bit for bit: a wire and an oblique slab (images to the end of the table), the
orthorhombic and the general grid of the Fortran-derived routines and the bulk grid of the default routines, Ge (POTRE)
and H with a core correction, one and several threads; the switch refuses an unknown value.
"""

from __future__ import annotations

import os
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the providers)
from parsec_python.acceleration.V_ion import spline_setup
from parsec_python.Grid.confined import build_confined_periodic_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion import ionic_potential as _ionic
from parsec_python.V_ion import load_pseudopotentials

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


_FOLDERS: list[Path] = []


def tearDownModule() -> None:
    import shutil

    for folder in _FOLDERS:
        shutil.rmtree(folder, ignore_errors=True)


def _confined_problem(kind: str):
    """The slab or wire of :mod:`.test_confined_setup` (FHI Si, H with a core correction), in a folder of this module."""

    import shutil
    import tempfile

    from parsec_python.acceleration.tests.test_confined_setup import _text
    from parsec_python.Input import parse_parsec_input

    folder = Path(tempfile.mkdtemp(prefix=f"spline_{kind}_"))
    _FOLDERS.append(folder)
    shutil.copy(DATA / "Si_FHIPP.DAT", folder / "Si_FHIPP.DAT")
    shutil.copy(DATA / "H_CORE_POTRE.DAT", folder / "H_POTRE.DAT")
    (folder / "parsec.in").write_text(_text(kind), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(folder / "parsec.in").problem


def _readable():
    return mock.patch.dict(os.environ, {spline_setup.SWITCH: "readable"})


def _germanium(spline: bool = True) -> dict:
    return {"Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 2, use_spline=spline)}


def _load(specifications):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_pseudopotentials(specifications)


def _cells():
    """(label, grid, atoms, specifications, lattice) of every grid kind."""

    cases = []
    cube = 7.2 * np.eye(3)
    fcc = 5.4 * (np.ones((3, 3)) - np.eye(3))
    atoms = [Atom("Ge", (0.31, 0.27, 0.40)), Atom("Ge", (1.83, 1.95, 1.62))]
    for label, lattice, fortran in (("fortran_cube", cube, True), ("fortran_fcc", fcc, True), ("default_cube", cube, False)):
        settings = PeriodicGridSettings(spacing=0.6, expansion_order=6, fortran_routines=fortran)
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)
        cases.append((label, grid, atoms, _germanium(), lattice))
    # Andrew Grieder's slab grid: its images along the two in-plane vectors (periodic_dimensions=2 of the readable sum)
    from parsec_python.Grid.pbc import build_slab_grid

    slab_cell = np.array([[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 9.0]])
    slab_grid = build_slab_grid(
        PeriodicCell(lattice_vectors=slab_cell, dimensions=2), PeriodicGridSettings(spacing=0.6, expansion_order=6)
    )
    slab_atoms = [Atom("Ge", (0.31, 0.27, 1.10)), Atom("Ge", (2.93, 2.85, -1.05))]
    cases.append(("default_slab", slab_grid, slab_atoms, _germanium(), slab_cell))
    for kind in ("slab", "wire"):
        problem = _confined_problem(kind)
        grid = build_confined_periodic_grid(problem.periodic_cell, problem.grid)
        specifications = dict(problem.pseudopotentials)
        specifications["H"] = replace(specifications["H"], use_spline=True)  # H carries the core correction
        cases.append((kind, grid, problem.atoms, specifications, problem.periodic_cell.lattice_vectors))
    return cases


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), (" Readable", "readable")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {spline_setup.SWITCH: value}):
                self.assertEqual(spline_setup.spline_setup_setting(), expected)
        with mock.patch.dict(os.environ, {spline_setup.SWITCH: "fast"}):
            with self.assertRaises(ValueError):
                spline_setup.spline_setup_setting()

    def test_what_it_takes(self) -> None:
        label, grid, atoms, specifications, lattice = _cells()[0]
        potentials = _load(specifications)
        self.assertIsNotNone(spline_setup.spline_superposed_density(grid, atoms, potentials, specifications, lattice, True))
        # a valence density, a linear core density, no lattice, the readable value
        self.assertIsNone(spline_setup.spline_superposed_density(grid, atoms, potentials, specifications, lattice, False))
        linear = _germanium(spline=False)
        self.assertIsNone(spline_setup.spline_superposed_density(grid, atoms, _load(linear), linear, lattice, True))
        self.assertIsNone(spline_setup.spline_superposed_density(grid, atoms, potentials, specifications, None, True))
        with _readable():
            self.assertIsNone(
                spline_setup.spline_superposed_density(grid, atoms, potentials, specifications, lattice, True)
            )


class DensityTests(unittest.TestCase):
    def test_the_spline_core_densities_are_the_readable_ones(self) -> None:
        for label, grid, atoms, specifications, lattice in _cells():
            potentials = _load(specifications)
            # the slab of the default routines: its caller (SCF.pbc) passes the two periodic dimensions
            dimensions = 2 if label == "default_slab" else 3
            with _readable():
                readable = _ionic.superpose_atomic_density(
                    grid, atoms, potentials, specifications, lattice, core=True, periodic_dimensions=dimensions
                )
            self.assertGreater(float(np.max(readable)), 0.0)
            for threads in ("1", "3"):
                with self.subTest(grid=label, threads=threads), mock.patch.dict(
                    os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}
                ), mock.patch.object(
                    spline_setup, "spline_superposed_density", wraps=spline_setup.spline_superposed_density
                ) as fast:
                    found = _ionic.superpose_atomic_density(
                        grid, atoms, potentials, specifications, lattice, core=True, periodic_dimensions=dimensions
                    )
                    self.assertGreaterEqual(fast.call_count, 1)
                    np.testing.assert_array_equal(found, readable)

    def test_a_prebuilt_spline_gives_the_same_values(self) -> None:
        potential = _load(_germanium())["Ge"]
        radii = np.linspace(0.0, 30.0, 20001)
        built = potential.density_spline(3, core=True)
        np.testing.assert_array_equal(
            potential.interpolate_density(radii, core=True, use_spline=True, spline_padding_width=3, spline=built),
            potential.interpolate_density(radii, core=True, use_spline=True, spline_padding_width=3),
        )


if __name__ == "__main__":
    unittest.main()
