"""The occupation an orbital needs to enter the exact exchange (``PARSEC_EXCHANGE_OCCUPATION_CUT``).

In a gap the Fermi-level bisection stops wherever the electron count is within
its tolerance (1e-10), so an empty level can keep an occupation between 1e-12
and 1e-10.  With ``on`` (default) such orbitals carry no exact exchange; with
``off`` the cut is 1e-12, as before.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python import cli as cli_module
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Occupations import fermi_occupations
from parsec_python.Occupations.fermi_dirac import FERMI_COUNT_TOLERANCE
from parsec_python.V_xc.exact_exchange import (
    EXCHANGE_OCCUPATION_CUT_ENV,
    OCCUPATION_THRESHOLD,
    exact_exchange_energy,
    exchange_occupation_cut_setting,
    exchange_occupation_threshold,
    occupied_columns,
    spin_exact_exchange_energy,
)
from parsec_python.V_xc.exchange_operator import (
    MultipolePairPoisson,
    apply_exchange_to_orbitals,
    compress_exchange,
    exchange_band_energy,
)
from parsec_python.models import GridSettings, HartreeSettings

DATA = Path(__file__).resolve().parents[1] / "data"


def _switch(value: str | None):
    environment = {key: item for key, item in os.environ.items() if key != EXCHANGE_OCCUPATION_CUT_ENV}
    if value is not None:
        environment[EXCHANGE_OCCUPATION_CUT_ENV] = value
    return mock.patch.dict(os.environ, environment, clear=True)


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with _switch(None):
            self.assertEqual(exchange_occupation_cut_setting(), "on")
            self.assertEqual(exchange_occupation_threshold(), FERMI_COUNT_TOLERANCE)
        with _switch(" ON "):
            self.assertEqual(exchange_occupation_threshold(), 1.0e-10)
        with _switch("off"):
            self.assertEqual(exchange_occupation_cut_setting(), "off")
            self.assertEqual(exchange_occupation_threshold(), OCCUPATION_THRESHOLD)
        with _switch("auto"), self.assertRaises(ValueError):
            exchange_occupation_cut_setting()

    def test_occupied_columns(self) -> None:
        occupations = np.array([1.0, 0.7, 2.0e-10, 5.0e-11, 2.0e-12, 1.0e-13, 0.0])
        with _switch(None):
            self.assertEqual(occupied_columns(occupations).tolist(), [0, 1, 2])
        with _switch("off"):
            self.assertEqual(occupied_columns(occupations).tolist(), [0, 1, 2, 3, 4])
        # An explicit threshold wins over the switch.
        with _switch(None):
            self.assertEqual(occupied_columns(occupations, 1.0e-12).tolist(), [0, 1, 2, 3, 4])

    def test_the_cut_leaves_out_what_a_gapped_bisection_leaves_on_the_empty_levels(self) -> None:
        # The converged levels of CO (PBE, 1160.45 K) of the core-level example: the bisection stops at
        # mu = -0.3494 Ry with 1.6e-12 and 1.4e-12 on the empty pair 27 kT above it.
        levels = np.array([
            -2.1398824130, -1.0364550731, -0.8632279908, -0.8630961588, -0.6660753672, -0.1498425462,
            -0.1485366185, 0.0386746657, 0.1238278309, 0.1402783253, 0.1571645774, 0.2287693773,
            0.2461171193, 0.2826481553, 0.2968879961, 0.3201591045,
        ])
        result = fermi_occupations(levels, 10.0, 1160.451812)
        self.assertAlmostEqual(result.fermi_level, -0.3494, places=4)
        self.assertTrue(np.all((result.occupations[5:7] > 1.0e-12) & (result.occupations[5:7] < 1.0e-10)))
        with _switch("off"):
            self.assertEqual(occupied_columns(result.occupations).size, 7)
        with _switch(None):
            self.assertEqual(occupied_columns(result.occupations).size, 5)


class ExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.6, radius=6.0, expansion_order=8))
        laplacian = build_negative_laplacian(cls.grid)
        cls.solver = MultipolePairPoisson(
            cls.grid, laplacian, HartreeSettings(multipole_order=6, relative_tolerance=1.0e-11, max_iterations=4000)
        )
        x = cls.grid.coordinates
        r2 = np.einsum("ij,ij->i", x, x)
        g = np.exp(-0.5 * r2)
        columns = [g, x[:, 0] * g, x[:, 1] * g, (x[:, 2] - 0.3) * np.exp(-0.3 * r2)]
        cls.orbitals, _ = np.linalg.qr(np.column_stack(columns))
        # Two full orbitals, an empty level at 4e-11 (below the count tolerance), one empty.
        cls.occupations = np.array([1.0, 1.0, 4.0e-11, 0.0])
        cls.volume = cls.grid.volume_element

    def solve(self, rho):
        return self.solver.solve_batch(rho[:, None])[:, 0]

    def test_energy_and_pair_solves(self) -> None:
        with _switch(None):
            cut = exact_exchange_energy(self.orbitals, self.occupations, self.volume, self.solve)
        with _switch("off"):
            full = exact_exchange_energy(self.orbitals, self.occupations, self.volume, self.solve)
        self.assertEqual(cut.orbitals, (0, 1))
        self.assertEqual(cut.pair_solves, 3)
        self.assertEqual(full.orbitals, (0, 1, 2))
        self.assertEqual(full.pair_solves, 6)
        # The level left out changes the energy by 2 f sum_j (2j|2j), far below 1e-10 Ry.
        self.assertLess(abs(cut.energy - full.energy), 1.0e-10)
        self.assertGreater(abs(cut.energy - full.energy), 0.0)

    def test_spin_channels(self) -> None:
        down = np.array([1.0, 6.0e-11, 0.0, 0.0])
        with _switch(None):
            cut = spin_exact_exchange_energy(
                (self.orbitals, self.orbitals), (self.occupations, down), self.volume, self.solve
            )
        with _switch("off"):
            full = spin_exact_exchange_energy(
                (self.orbitals, self.orbitals), (self.occupations, down), self.volume, self.solve
            )
        self.assertEqual(cut.orbitals, (0, 1, 4))
        self.assertEqual(cut.pair_solves, 3 + 1)
        self.assertEqual(full.orbitals, (0, 1, 2, 4, 5))
        self.assertEqual(full.pair_solves, 6 + 3)
        self.assertLess(abs(cut.energy - full.energy), 1.0e-10)

    def test_operator_and_band_energy(self) -> None:
        with _switch(None):
            cut = apply_exchange_to_orbitals(self.orbitals, self.occupations, self.volume, self.solver)
            operator = compress_exchange(self.orbitals, cut, fraction=0.25)
            band = exchange_band_energy(operator, self.orbitals, self.occupations)
        with _switch("off"):
            full = apply_exchange_to_orbitals(self.orbitals, self.occupations, self.volume, self.solver)
            operator_full = compress_exchange(self.orbitals, full, fraction=0.25)
            band_full = exchange_band_energy(operator_full, self.orbitals, self.occupations)
        self.assertEqual(cut.occupied, (0, 1))
        self.assertEqual(full.occupied, (0, 1, 2))
        # Occupied pairs, then the two other columns with each occupied source.
        self.assertEqual(cut.pair_solves, 3 + 2 * 2)
        self.assertEqual(full.pair_solves, 6 + 1 * 3)
        self.assertLess(abs(cut.energy - full.energy), 1.0e-10)
        self.assertLess(abs(band - band_full), 1.0e-10)
        scale = float(np.max(np.abs(full.applied)))
        self.assertLess(float(np.max(np.abs(cut.applied - full.applied))) / scale, 1.0e-9)


class CommandLineTests(unittest.TestCase):
    INPUT = """Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.6 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 8
States_Num: 2
Fermi_Temp: 80
Max_Iter: 60
Convergence_Criterion: 1.0d-6 Ry
Eigensolver: chebff
Subspace_Buffer_Size: 6
Mixing_Method: Anderson
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
Read_VCD: false
Atomic_Energy_Correction: -0.25 Ry
begin Atom_Coord
    0.0  0.0  0.7
    0.0  0.0 -0.7
end Atom_Coord
Correlation_Type: ca
"""

    def test_an_unknown_value_is_refused_before_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            path = directory / "parsec.in"
            path.write_text(self.INPUT)
            errors = StringIO()
            with _switch("maybe"), redirect_stdout(StringIO()), redirect_stderr(errors):
                code = cli_module.main([str(path), "--quiet", "--energy-refinement", "b3lyp"])
            self.assertEqual(code, 2)
            self.assertIn(EXCHANGE_OCCUPATION_CUT_ENV, errors.getvalue())
            self.assertFalse((directory / "parsec.out").exists())


if __name__ == "__main__":
    unittest.main()
