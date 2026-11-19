"""Small fixes of 2026-11-18 found by the survey of how the SCF uses symmetry (stream X).

* The count of the sector states that the storage and margin rules read
  (``CuPySymmetrySCFEigensolver._state_fit_bytes``) holds the columns of both
  channels of a spin-polarized loop on an MPI rank that shares a sector basis
  with its group, as it does on one device.

* A cluster with a point group whose Hartree solve (and CA-LDA) leaves the
  sectors for the full grid for want of the native extension says so in a
  NOTE line of parsec.out (``driver._cluster_full_grid_notes``).
* The MPI runner says in a NOTE when ranks share sectors after a CHEBDAV
  first solve, which gives each of them the whole basis
  (``mpi_full_scf._shared_first_solve_note``).
* A ``relax_restart.dat`` that holds an atom of a cell on the default
  routines at -1e-9 bohr (on a face to round-off) is prepared, folded into
  the cell for the builders (``SCF.pbc.atoms_for_cell_builders``).

All tests run on the CPU: the count reads no device for a shared group.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


class SharedGroupSpinCountTests(unittest.TestCase):
    """``_state_fit_bytes`` of a sector whose basis the MPI ranks of a group share."""

    rows = 1_000_003
    count = 1_234

    def counted(self, *, channels: int, route: str, initial_method: str = "chebff", size: int = 4) -> int:
        from parsec_python.acceleration.Eigensolvers import symmetry

        solver = SimpleNamespace(settings=SimpleNamespace(initial_method=initial_method))
        group = SimpleNamespace(size=size, ritz_route=route)
        stub = SimpleNamespace(
            resident_device_bytes={},
            device_ids=(0,),
            _solvers=[solver],
            decomposition=SimpleNamespace(sector_size=lambda representation: self.rows),
            _sector_device_ids=[0],
            _operators=[SimpleNamespace(_mpi_shared_group=group)],
            spin_channel_count=channels,
        )
        with patch.object(symmetry, "require_cupy", return_value=(None, None)):
            return symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes(stub, [self.count])[0]

    def test_every_shared_route_counts_the_columns_of_both_channels(self) -> None:
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import exchange_bytes

        share = -(-self.count // 4)
        columns = self.rows * share * 8
        for route in ("block", "rounds", "rows"):
            with self.subTest(route=route):
                one = self.counted(channels=1, route=route)
                two = self.counted(channels=2, route=route)
                # The second channel adds this rank's columns of its basis, and nothing else.
                self.assertEqual(two - one, columns)
        # The unpolarized counts are those of before.
        self.assertEqual(self.counted(channels=1, route="block"),
                         columns + 4 * exchange_bytes() + 11 * self.count ** 2 * 8)
        self.assertEqual(self.counted(channels=1, route="rounds"),
                         2 * columns + 3 * exchange_bytes() + 2 * self.count ** 2 * 8)
        self.assertEqual(self.counted(channels=1, route="rows"), 4 * columns + 2 * exchange_bytes())

    def test_a_whole_basis_after_chebdav_counts_both_channels_as_before(self) -> None:
        one = self.counted(channels=1, route="block", initial_method="chebdav")
        two = self.counted(channels=2, route="block", initial_method="chebdav")
        self.assertEqual(two - one, self.rows * self.count * 8)


class ClusterFullGridNoteTests(unittest.TestCase):
    """The NOTE of a cluster whose Hartree solve or CA-LDA leaves the sectors without the native extension."""

    @staticmethod
    def selection(requested, hartree, reasons=()):
        return SimpleNamespace(requested=requested, selected="cupy", hartree_backend=hartree,
                               fallback_reasons=tuple(reasons))

    def test_the_note_names_the_full_grid_and_the_reason(self) -> None:
        from parsec_python.acceleration import driver

        notes = driver._cluster_full_grid_notes(self.selection("cupy", "cupy"), periodic=False, group_found=True,
                                                orbitals_in_sectors=True)
        self.assertEqual(len(notes), 1)
        for words in ("Hartree solve of this cluster runs on the full grid", "--backend cupy",
                      "the orbitals stay in the symmetry sectors"):
            self.assertIn(words, notes[0])
        (missing,) = driver._cluster_full_grid_notes(
            self.selection("auto", "cupy", ["native components skipped: no module named _parsec_native"]),
            periodic=False, group_found=True, orbitals_in_sectors=False)
        self.assertIn("the native C++ extension is not available (native components skipped", missing)
        self.assertNotIn("orbitals stay", missing)
        xc = driver._cluster_xc_full_grid_note(self.selection("cupy", "cupy"), functional="CA-LDA")
        self.assertIn("XC functional (CA-LDA) of this cluster is evaluated on the full grid of the host", xc)

    def test_no_note_where_the_wedge_serves_or_nothing_is_lost(self) -> None:
        from parsec_python.acceleration import driver

        for selection, periodic, group in ((self.selection("auto", "native"), False, True),
                                           (self.selection("cupy", "cupy"), True, True),
                                           (self.selection("cupy", "cupy"), False, False)):
            with self.subTest(hartree=selection.hartree_backend, periodic=periodic, group=group):
                self.assertEqual(driver._cluster_full_grid_notes(selection, periodic=periodic, group_found=group,
                                                                 orbitals_in_sectors=group), [])

    def test_the_report_writes_the_note_at_every_output_level(self) -> None:
        from parsec_python.acceleration.Output.accelerated_output import AcceleratedTextReporter

        written = []
        reporter = SimpleNamespace(reference=SimpleNamespace(setup=lambda system: None),
                                   _start_radius_estimates=lambda system: None, output_level=1,
                                   write=written.append)
        details = (("orbital_symmetry", "CuPy real sectors"), ("symmetry_full_grid_note", "the Hartree solve ..."),
                   ("hartree_backend", "cupy"))
        system = SimpleNamespace(
            backend_info=SimpleNamespace(details=details, requested="cupy", selected="cupy", dtype="float64",
                                         device="cuda:0", implementation="x", fallback_reasons=()),
            backend=SimpleNamespace(statistics=SimpleNamespace(initialization_seconds=0.0)))
        AcceleratedTextReporter.setup(reporter, system)
        text = "\n".join(written)
        self.assertIn(" NOTE: the Hartree solve ...", text)
        self.assertNotIn("symmetry_full_grid_note =", text)
        self.assertIn(" hartree_backend = cupy", text)


class SharedChebdavNoteTests(unittest.TestCase):
    """The NOTE of the MPI runner where ranks share sectors after a CHEBDAV first solve."""

    def test_the_note_recommends_chebff(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        context = SimpleNamespace(shared=True, rank_groups=((0, 1), (2, 3), (4, 5, 6)))
        note = mpi_full_scf._shared_first_solve_note(context, "chebdav")
        self.assertTrue(note.startswith("NOTE: 3 sector(s) are shared by groups of 2/3 MPI ranks"))
        self.assertIn("Eigensolver: chebff divides each basis", note)
        one_sector = SimpleNamespace(shared=True, rank_groups=((0, 1, 2, 3),))
        self.assertIn("groups of 4 MPI ranks", mpi_full_scf._shared_first_solve_note(one_sector, "chebdav"))

    def test_no_note_for_chebff_or_unshared_sectors(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        shared = SimpleNamespace(shared=True, rank_groups=((0, 1),))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(shared, "chebff"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(shared, "CHEBFF"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(None, "chebdav"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(
            SimpleNamespace(shared=False, rank_groups=((0,), (1,))), "chebdav"))


_SILICON = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
 10.26 10.26 10.26
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.75
Expansion_Order: 6
States_Num: 26
Fermi_Temp: 500 K
Correlation_Type: ca
Eigensolver: chebff
Minimization: simple
Movement_Num: 1
Relax_Restart: .true.
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
 0.00 0.00 0.00
 0.00 0.50 0.50
 0.50 0.00 0.50
 0.50 0.50 0.00
 0.25 0.25 0.25
 0.25 0.75 0.75
 0.75 0.25 0.75
 0.75 0.75 0.25
end Atom_Coord
"""


class RestartOnCellFaceTests(unittest.TestCase):
    """A ``relax_restart.dat`` of a cell on the default routines with an atom at -1e-9 bohr."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        shutil.copy(DATA / "Si_POTRE.DAT", self.directory / "Si_POTRE.DAT")
        (self.directory / "parsec.in").write_text(_SILICON, encoding="utf-8")

    def restarted(self, first):
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.Relaxation.files import RelaxRestart, format_relax_restart
        from parsec_python.Relaxation.run import restart_structure

        translation = parse_parsec_input(self.directory / "parsec.in")
        self.assertFalse(translation.problem.grid.fortran_routines)
        positions = np.array([atom.position for atom in translation.problem.atoms], dtype=np.float64)
        positions[0] = first
        (self.directory / "relax_restart.dat").write_text(
            format_relax_restart(RelaxRestart(positions=positions, iteration=1, movable_atoms=8)), encoding="utf-8")
        restarted, _restart, notes = restart_structure(translation, self.directory)
        self.assertTrue(any("relax_restart.dat" in note for note in notes))
        return translation, restarted

    def test_an_atom_on_a_face_is_prepared_folded_for_the_builders(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF.pbc import atoms_for_cell_builders, prepare_periodic_single_point

        original, restarted = self.restarted([-1.0e-9, 0.0, -1.1e-9])
        atoms = restarted.problem.atoms
        np.testing.assert_array_equal(atoms[0].position, [-1.0e-9, 0.0, -1.1e-9])
        lattice = restarted.problem.periodic_cell.lattice_vectors
        folded, beyond = atoms_for_cell_builders(lattice, atoms)
        self.assertIsNone(beyond)
        self.assertTrue(np.all(np.asarray(folded[0].position) >= 0.0))
        self.assertTrue(all(new is old for new, old in zip(folded[1:], atoms[1:])))
        # Inside the cell nothing changes: the same atoms, bit for bit.
        inside = original.problem.atoms
        self.assertIs(atoms_for_cell_builders(lattice, inside)[0][0], inside[0])
        system = prepare_periodic_single_point(restarted.problem)
        # The system keeps the coordinates of the file (an optimizer continues from them); its fields are those of
        # the atoms folded into the cell, and within round-off those of the atom at the corner.
        np.testing.assert_array_equal(system.atoms[0].position, atoms[0].position)
        at_folded = prepare_periodic_single_point(replace(restarted.problem, atoms=folded))
        np.testing.assert_array_equal(system.ionic_potential, at_folded.ionic_potential)
        self.assertEqual(system.ion_ion_energy, at_folded.ion_ion_energy)
        at_corner = prepare_periodic_single_point(original.problem)
        np.testing.assert_allclose(system.ionic_potential, at_corner.ionic_potential, rtol=0.0, atol=1.0e-6)
        self.assertAlmostEqual(system.ion_ion_energy, at_corner.ion_ion_energy, delta=1.0e-6)

    def test_an_atom_farther_outside_is_still_refused(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        _original, restarted = self.restarted([-1.0e-3, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "lies outside the periodic cell"):
            prepare_periodic_single_point(restarted.problem)


if __name__ == "__main__":
    unittest.main()
