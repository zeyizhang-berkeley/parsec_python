"""Units of irreducible representations on partner rows in complete GPU calculations (PARSEC_ORBITAL_IRREP_ROWS).

The A1 and A2 units of a Td Ge5 cluster are the sectors A1 and A2 of Td itself: on ``partner`` (the default since
2026-11-22; ``sector`` keeps the rows of before) they are solved on the orbits of Td with stencils and projectors of
their own and no projector after the filters, three times fewer rows than the sector of D2d.  The SCF energies, the
occupied levels, the forces and the exported orbitals (the occupied subspace on the full grid) equal those of the units
on the rows of their sectors (``sector``) and of the weighted sectors of before (``PARSEC_ORBITAL_IRREPS=off``), on one
GPU and on the GPUs of a node (units placed by cost), on a coarse grid with a high first-filter degree, and for the Si8
cube with its 96 and 192 operations.  LDA+U, which reads the sector orbitals on the rows of their sectors, keeps those
rows under the default (a Td Ag4 cluster with U, CHEBFF and CHEBDAV).
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.tests.test_orbital_irreps_device import _available, _ge5, _run

_PARTNER = {"PARSEC_ORBITAL_IRREP_ROWS": "partner"}
_SECTOR = {"PARSEC_ORBITAL_IRREP_ROWS": "sector"}


def _ag4u(directory: Path, method: str = "chebff"):
    """A Td Ag4 cluster with LDA+U (U = 4 eV on Ag), the weighted sectors of PARSEC_LDAU_WEIGHTED_SECTORS."""

    from parsec_python.Input import parse_parsec_input

    data = Path(__file__).resolve().parents[2] / "tests" / "data"
    corners = 2.75 * np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=float)
    lines = [
        "Boundary_Conditions: cluster", "Boundary_Sphere_Radius: 9.0", "Grid_Spacing: 0.7",
        "Coordinate_Unit: Cartesian_Bohr", "States_Num: 34", "Correlation_Type: ca", "Convergence_Criterion: 1.0e-8",
        "Max_Iter: 200", f"Eigensolver: {method}", "Mixing_Param: 0.2", "Fermi_Temp: 1000", "Atom_Types_Num: 1",
        "Atom_Type: Ag", "Local_Component: s", "ldaplusu_u 4.0 eV", "begin Atom_Coord",
        *(f"  {x:.10f} {y:.10f} {z:.10f}" for x, y, z in corners), "end Atom_Coord",
    ]
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    (directory / "Ag_POTRE.DAT").write_bytes((data / "Ag_POTRE.DAT").read_bytes())
    return parse_parsec_input(directory / "parsec.in").problem


def _occupied_overlap(left, right) -> np.ndarray:
    """The singular values of the overlap of the fully occupied exported orbitals of two results."""

    def occupied(result):
        wavefunctions = np.asarray(result.wavefunctions, dtype=np.float64)
        columns = np.flatnonzero(np.asarray(result.occupations) > 1.0 - 1.0e-8)
        return wavefunctions[:, columns]

    a, b = occupied(left), occupied(right)
    if a.shape != b.shape:
        raise AssertionError(f"occupied orbitals of shapes {a.shape} and {b.shape}")
    return np.linalg.svd(a.T @ b, compute_uv=False), a


@unittest.skipUnless(_available(), "CUDA runtime or native extension is unavailable")
class PartnerRowSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.ge5 = _ge5(Path(directory.name))

    def assert_same(self, left, right, label, *, energy=2.0e-8, level=1.0e-7, force=2.0e-6) -> None:
        (result, forces, _, _), (reference, reference_forces, _, _) = left, right
        self.assertTrue(result.converged and reference.converged, label)
        for field, delta in (("total", energy), ("exchange_correlation", energy), ("hartree", energy),
                             ("eigenvalue", level)):
            self.assertAlmostEqual(
                getattr(result.energies, field), getattr(reference.energies, field), delta=delta, msg=f"{label} {field}"
            )
        occupied = int(np.count_nonzero(np.asarray(reference.occupations) > 1.0e-6))
        np.testing.assert_allclose(
            np.sort(result.eigenvalues)[:occupied], np.sort(reference.eigenvalues)[:occupied], rtol=0, atol=level,
            err_msg=label,
        )
        if forces is not None and reference_forces is not None:
            np.testing.assert_allclose(forces.total, reference_forces.total, rtol=0, atol=force, err_msg=label)

    def check_partner_sectors(self, run, units: int) -> None:
        sectors, details = run[2], run[3]
        decomposition = sectors.decomposition
        partner = decomposition.partner_units
        self.assertEqual(len(partner), units)
        for index in partner:
            operator = sectors._operators[index]
            if operator is None:
                continue
            self.assertIsNone(getattr(operator, "irrep_projection", None))
            self.assertEqual(int(operator.effective_potential.size), decomposition.solved_size(index))
            self.assertLess(decomposition.solved_size(index), decomposition.sector_size(index))
        for index in decomposition.projected_sectors:
            operator = sectors._operators[index]
            if index not in partner and operator is not None:
                self.assertIsNotNone(getattr(operator, "irrep_projection", None))
        self.assertIn("on the orbits of a larger subgroup", details.get("orbital_partner_rows", ""))
        self.assertIn(f"{units} of them on the orbits of larger subgroups", details["symmetry_detection"])

    def test_a_td_cluster_on_one_gpu_and_on_a_node(self) -> None:
        one = {"PARSEC_CUPY_DEVICES": "current"}
        before = _run(self.ge5, dict(one, PARSEC_ORBITAL_IRREPS="off"))
        for mode in ("once", "split"):
            with self.subTest(mode=mode):
                rows = _run(self.ge5, dict(one, PARSEC_ORBITAL_IRREPS=mode, **_SECTOR))
                partner = _run(self.ge5, dict(one, PARSEC_ORBITAL_IRREPS=mode, **_PARTNER))
                self.check_partner_sectors(partner, 2)
                self.assertEqual(rows[2]._partner_units, ())
                self.assert_same(partner, rows, f"Ge5 {mode} partner against sector rows")
                self.assert_same(partner, before, f"Ge5 {mode} partner against off")
        # all the GPUs of the node (the units placed by cost)
        node = _run(self.ge5, dict(_PARTNER, PARSEC_ORBITAL_IRREPS="once"))
        self.check_partner_sectors(node, 2)
        self.assert_same(node, before, "Ge5 once partner, the GPUs of the node")

    def test_lda_u_keeps_the_rows_of_the_sectors(self) -> None:
        # LDA+U forms its occupation matrices from the sector orbitals on the rows of their sectors: under the partner
        # default the units keep those rows (driver._prepare_single_point, a NOTE in the details) and give the energies,
        # levels and forces of the weighted sectors of before; until 2026-11-22 this run stopped in the first LDA+U
        # occupation matrices (hubbard_device: units on partner rows are not supported)
        one = {"PARSEC_CUPY_DEVICES": "current", "PARSEC_ORBITAL_IRREP_ROWS": ""}
        for method in ("chebff", "chebdav"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                problem = _ag4u(Path(directory), method)
                before = _run(problem, dict(one, PARSEC_ORBITAL_IRREPS="off"))
                units = _run(problem, dict(one, PARSEC_ORBITAL_IRREPS="once"))
                decomposition = units[2].decomposition
                self.assertTrue(decomposition.projected_sectors)
                self.assertTrue(decomposition.reduction.sector_partner_cosets)
                self.assertEqual(decomposition.partner_units, ())
                self.assertEqual(decomposition.sector_rows_kept, "LDA+U")
                self.assertEqual(units[2]._partner_units, ())
                notes = [value for key, value in units[0].backend.details if key == "symmetry_full_grid_note"]
                self.assertTrue(any("LDA+U keeps the units" in note for note in notes), notes)
                self.assertNotIn("on the orbits of larger subgroups", units[3]["symmetry_detection"])
                # the total energy within 1e-7 Ry at the criterion of 1e-8 Ry; its Hartree and exchange-correlation
                # terms, which the criterion leaves free to first order, within 5e-7 Ry (1.05e-7 apart on A100)
                self.assertAlmostEqual(units[0].energies.total, before[0].energies.total, delta=1.0e-7)
                self.assert_same(units, before, f"Ag4+U {method} once against off", energy=5.0e-7, level=1.0e-6,
                                 force=1.0e-5)

    def test_the_exported_orbitals(self) -> None:
        one = {"PARSEC_CUPY_DEVICES": "current", "PARSEC_ORBITAL_IRREPS": "once"}

        def exported(environment):
            from parsec_python.acceleration import driver

            with mock.patch.dict(os.environ, environment):
                system = driver.prepare_single_point(self.ge5)
                return driver.run_scf(system)

        partner = exported(dict(one, **_PARTNER))
        rows = exported(dict(one, **_SECTOR))
        values, a = _occupied_overlap(partner, rows)
        np.testing.assert_allclose(values, np.ones_like(values), rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(a.T @ a, np.eye(a.shape[1]), rtol=0, atol=1.0e-10)
        # the density of the exported orbitals is that of the SCF on the full grid
        density = np.asarray(partner.wavefunctions) ** 2 @ np.asarray(partner.occupations)
        reference = np.asarray(rows.wavefunctions) ** 2 @ np.asarray(rows.occupations)
        scale = float(np.max(np.abs(reference)))
        self.assertLess(float(np.max(np.abs(density - reference))), 1.0e-6 * scale)

    def test_a_coarse_grid_with_a_high_first_filter_degree(self) -> None:
        once = {"PARSEC_ORBITAL_IRREPS": "once", "PARSEC_CUPY_DEVICES": "current"}
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(radius=9.5, states=24, spacing=0.7, first_filter_cycles=4)
            problem = _ge5(Path(directory), first_filter_degree=90, **settings)
            before = _run(problem, {"PARSEC_ORBITAL_IRREPS": "off", "PARSEC_CUPY_DEVICES": "current"})
            partner = _run(problem, dict(once, **_PARTNER))
        self.check_partner_sectors(partner, 2)
        # as the units on the rows of their sectors (test_orbital_irreps_device): the total energy, the occupied
        # levels and the forces of off
        self.assertTrue(partner[0].converged and before[0].converged)
        self.assertAlmostEqual(partner[0].energies.total, before[0].energies.total, delta=1.0e-9)
        occupied = int(np.count_nonzero(np.asarray(before[0].occupations) > 1.0e-6))
        np.testing.assert_allclose(np.sort(partner[0].eigenvalues)[:occupied],
                                   np.sort(before[0].eigenvalues)[:occupied], rtol=0, atol=5.0e-8)
        np.testing.assert_allclose(partner[1].total, before[1].total, rtol=0, atol=1.0e-8)

    def test_silicon_cells(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = silicon("pbe")
        problem = replace(problem, eigensolver=replace(problem.eigensolver, method="chebff"))
        one = {"PARSEC_CUPY_DEVICES": "current"}
        for translations, mode in (("off", "once"), ("auto", "split")):
            environment = dict(one, PARSEC_PERIODIC_TRANSLATIONS=translations)
            with self.subTest(translations=translations, mode=mode):
                before = _run(problem, dict(environment, PARSEC_ORBITAL_IRREPS="off"), forces=False, backend="cupy")
                rows = _run(problem, dict(environment, PARSEC_ORBITAL_IRREPS=mode, **_SECTOR), forces=False,
                            backend="cupy")
                partner = _run(problem, dict(environment, PARSEC_ORBITAL_IRREPS=mode, **_PARTNER), forces=False,
                               backend="cupy")
                self.assertTrue(partner[2].decomposition.partner_units, f"{translations} {mode}")
                self.assert_same(partner, rows, f"Si8 {translations} {mode} partner against sector rows")
                self.assert_same(partner, before, f"Si8 {translations} {mode} partner against off")


if __name__ == "__main__":
    unittest.main()
