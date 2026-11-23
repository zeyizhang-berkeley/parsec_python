"""The spin-orbit projectors of a ``Cubic_Spline`` species on PARSEC's spline (``PARSEC_SOC_PROJECTOR_SPLINE``).

``nonloc.F90`` evaluates ``vionr`` and ``vsor`` with ``splint`` under ``uspline``, as it does the Kleinman-Bylander
projectors; the port interpolated them linearly for every species.  The energy of linear tables has a kink wherever a
grid point crosses a radial knot, so their forces, exact derivatives over steps of 1e-6 bohr, miss the slope of the
energy over the steps of a relaxation.  Checked here with fixed spinors: under ``auto`` the projectors of a
``Cubic_Spline`` species are the spline's values and the spin-orbit forces of both gradient builders follow the energy
over steps of 1e-3 and 2e-3 bohr; ``off`` and a species without ``Cubic_Spline`` keep the linear tables bit for bit;
an unknown value is refused.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Eigensolvers.perturbative_soc import (
    SOC_PROJECTOR_SPLINE_ENV,
    build_spin_orbit_projectors,
    soc_projector_spline_setting,
)
from parsec_python.Forces import spin_orbit as default_forces
from parsec_python.Forces import spin_orbit_fortran as fortran_forces
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Pseudopotential.radial_spline import ParsecRadialSpline
from parsec_python.V_ion import load_pseudopotentials

_ROOT = Path(__file__).resolve().parents[4]
_FILES = {
    "Au": (_ROOT / "examples" / "0d_AuH" / "Au_POTRE.DAT", 0),
    "Ge": (_ROOT / "src" / "parsec_python" / "tests" / "data" / "Ge_POTRE.DAT", 2),
}


def _setup(symbol: str, spline: bool):
    path, local = _FILES[symbol]
    species = {symbol: SpeciesPotential(path, local, spin_orbit=True, use_spline=spline)}
    atoms = (Atom(symbol, np.array([3.413, 3.127, 3.291])), Atom(symbol, np.array([0.61, 3.52, 2.93])))
    cell = PeriodicCell(lattice_vectors=8.0 * np.eye(3))
    grid = build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=0.4, expansion_order=8))
    potentials = load_pseudopotentials(species)
    rng = np.random.default_rng(1)
    envelope = np.exp(
        -np.min(np.stack([np.sum((grid.coordinates - a.position) ** 2, axis=1) for a in atoms]), axis=0) / 4.0
    )
    spinors = (rng.normal(size=(2 * grid.size, 4)) + 1j * rng.normal(size=(2 * grid.size, 4))) * np.tile(
        envelope, 2
    )[:, None]
    spinors /= np.linalg.norm(spinors, axis=0)
    return species, atoms, cell.lattice_vectors, grid, potentials, spinors, np.array([1.0, 0.7, 0.3, 0.05])


def _moved(atoms, axis, step):
    out = list(atoms)
    position = np.array(out[0].position, dtype=float)
    position[axis] += step
    out[0] = Atom(out[0].symbol, position)
    return tuple(out)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(SOC_PROJECTOR_SPLINE_ENV, None)
            self.assertEqual(soc_projector_spline_setting(), "auto")
        with mock.patch.dict(os.environ, {SOC_PROJECTOR_SPLINE_ENV: " OFF "}):
            self.assertEqual(soc_projector_spline_setting(), "off")
        with mock.patch.dict(os.environ, {SOC_PROJECTOR_SPLINE_ENV: "linear"}):
            with self.assertRaisesRegex(ValueError, "auto or off"):
                soc_projector_spline_setting()


class ProjectorTests(unittest.TestCase):
    def table_values(self, symbol: str, spline: bool, setting: str) -> tuple[np.ndarray, np.ndarray]:
        species, atoms, lattice, grid, potentials, _spinors, _occupations = _setup(symbol, spline)
        with mock.patch.dict(os.environ, {SOC_PROJECTOR_SPLINE_ENV: setting}):
            projectors = build_spin_orbit_projectors(grid, atoms, potentials, species, lattice, None)
        potential = potentials[symbol]
        atom = projectors[0]
        relative = grid.coordinates[atom.support_rows] - atoms[0].position
        radius = np.linalg.norm(relative, axis=1)
        local = species[symbol].local_angular_momentum
        l = 1 if 1 in potential.spin_orbit_channel_potentials and local != 1 else 2
        radial_ion, radial_so, _ = potential.spin_orbit_projectors(l, local)
        column = {1: 0, 2: 3}[l]
        # |v_so| over the 2l+1 harmonics: sqrt(dV) |R(r)| sqrt((2l+1)/(4 pi)) (sum of |Y_lm|^2)
        found = np.sqrt(np.sum(np.abs(atom.v_so[:, column:column + 2 * l + 1]) ** 2, axis=1))
        scale = np.sqrt(grid.volume_element) * np.sqrt((2 * l + 1) / (4.0 * np.pi))
        if spline and setting == "auto":
            expected = np.abs(
                ParsecRadialSpline.from_positive_grid(potential.radii, radial_so, grid.settings.stencil_half_width)(
                    np.maximum(radius, potential.radii[0])
                )
            )
        else:
            expected = np.abs(np.interp(radius, potential.radii, radial_so, left=radial_so[0], right=0.0))
        return found, expected * scale

    def test_cubic_spline_species_take_the_spline(self) -> None:
        for symbol in ("Au", "Ge"):
            with self.subTest(symbol=symbol):
                found, expected = self.table_values(symbol, True, "auto")
                np.testing.assert_allclose(found, expected, rtol=1.0e-12, atol=1.0e-14)

    def test_off_and_linear_species_keep_the_linear_tables(self) -> None:
        for spline, setting in ((True, "off"), (False, "auto"), (False, "off")):
            with self.subTest(spline=spline, setting=setting):
                found, expected = self.table_values("Au", spline, setting)
                np.testing.assert_allclose(found, expected, rtol=1.0e-12, atol=1.0e-14)
        # off restores the projectors of the linear tables bit for bit
        species, atoms, lattice, grid, potentials, _s, _o = _setup("Au", True)
        with mock.patch.dict(os.environ, {SOC_PROJECTOR_SPLINE_ENV: "off"}):
            off = build_spin_orbit_projectors(grid, atoms, potentials, species, lattice, None)
        linear_species = {"Au": SpeciesPotential(_FILES["Au"][0], 0, spin_orbit=True, use_spline=False)}
        linear = build_spin_orbit_projectors(grid, atoms, potentials, linear_species, lattice, None)
        for a, b in zip(off, linear):
            np.testing.assert_array_equal(a.v_ion, b.v_ion)
            np.testing.assert_array_equal(a.v_so, b.v_so)


class ForceTests(unittest.TestCase):
    """The spin-orbit forces of fixed spinors against the slope of their energy at the steps of a relaxation."""

    def slopes(self, setting: str, symbol: str = "Au") -> tuple[np.ndarray, np.ndarray, list]:
        species, atoms, lattice, grid, potentials, spinors, occupations = _setup(symbol, True)
        with mock.patch.dict(os.environ, {SOC_PROJECTOR_SPLINE_ENV: setting}):
            projectors = build_spin_orbit_projectors(grid, atoms, potentials, species, lattice, None)
            gradients = fortran_forces.build_spin_orbit_projector_gradients(
                grid, atoms, potentials, species, lattice, None
            )
            fortran = fortran_forces.spin_orbit_forces(projectors, gradients, spinors, occupations, 2)
            operator = default_forces.build_spin_orbit_force_operator(
                grid, atoms, potentials, species, projectors, lattice, None
            )
            default = default_forces.spin_orbit_forces(operator, spinors, occupations, 2)
            numeric = []
            for axis in range(3):
                row = []
                for step in (1.0e-3, 2.0e-3):
                    energies = [
                        fortran_forces.spin_orbit_energy(
                            build_spin_orbit_projectors(
                                grid, _moved(atoms, axis, sign * step), potentials, species, lattice, None
                            ),
                            spinors,
                            occupations,
                        )
                        for sign in (1.0, -1.0)
                    ]
                    row.append(-(energies[0] - energies[1]) / (2.0 * step))
                numeric.append(row)
        return fortran, default, numeric

    def test_the_spline_forces_follow_the_energy(self) -> None:
        for symbol in ("Au", "Ge"):
            with self.subTest(symbol=symbol):
                fortran, default, numeric = self.slopes("auto", symbol)
                np.testing.assert_allclose(default, fortran, atol=1.0e-12)
                scale = float(np.max(np.abs(fortran[0])))
                for axis in range(3):
                    for slope in numeric[axis]:
                        self.assertLess(abs(fortran[0, axis] - slope), 2.0e-4 * scale + 1.0e-10)

    def test_the_linear_forces_miss_the_slope(self) -> None:
        # (the defect the spline removes: linear tables, the same spinors, the same steps)
        fortran, _default, numeric = self.slopes("off", "Au")
        worst = max(abs(fortran[0, axis] - slope) for axis in range(3) for slope in numeric[axis])
        self.assertGreater(worst, 1.0e-3 * float(np.max(np.abs(fortran[0]))))


if __name__ == "__main__":
    unittest.main()
