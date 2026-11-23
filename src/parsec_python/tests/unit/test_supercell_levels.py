"""The levels of every k-point of a mesh solved as its supercell (``SCF/supercell_levels.py``).

CPU: states of a supercell made from Bloch functions of the 2 x 2 x 2 mesh of the silicon cube (each a periodic
factor times the +-1 phases of the cells), some degenerate levels mixed over the k-points of their star.  The
phases of the translations give every state its k-point, the clusters that mix them their k-points as a set; the
levels, weights and occupations of ``parsec.dat`` and the archive of the cell are those of the k-point route's
irreducible wedge; the characters of sectors whose group holds the translations, and the partners of weighted
states over the star of their k-point.  The A100 check against the k-point route is in
``acceleration/tests/test_kpoint_supercell_levels.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.SCF import kpoint_supercell as route
from parsec_python.SCF import supercell_levels as levels
from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input, _translation


def _bits_of(index: int) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.unravel_index(index, (2, 2, 2)))


def _bloch_states(cell_counts, rng, per_point: int, clusters=()):
    """Real Bloch states of the 2 x 2 x 2 mesh on the supercell grid: ``(vectors, eigenvalues, points)``.

    Every mesh point holds ``per_point`` levels; the levels of a star (points of the same number of half phases,
    the stars of the simple cubic mesh: Gamma, X, M, R) are equal.  ``clusters``: indices of levels whose states
    of the whole star are mixed by a random rotation (a degenerate level a solver returns in any basis)."""

    cell_counts = tuple(int(value) for value in cell_counts)
    supercounts = tuple(2 * value for value in cell_counts)
    cell_index = np.stack(
        np.meshgrid(*[np.arange(n) // c for n, c in zip(supercounts, cell_counts)], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    star_levels = {stars: np.sort(rng.normal(size=per_point)) + 3.0 * stars for stars in range(4)}
    columns, values, points = [], [], []
    for level in range(per_point):
        group_columns = []
        for point in range(8):
            bits = np.asarray(_bits_of(point))
            periodic = rng.normal(size=cell_counts)
            state = np.tile(periodic, (2, 2, 2)).reshape(-1) * np.where(cell_index @ bits % 2, -1.0, 1.0)
            columns.append(state / np.linalg.norm(state))
            values.append(star_levels[int(bits.sum())][level])
            points.append(point)
            group_columns.append(len(columns) - 1)
        if level in clusters:
            # mix every state of this level that shares its eigenvalue (the members of one star)
            for stars in range(4):
                members = [index for index in group_columns if bin(points[index]).count("1") == stars]
                block = np.column_stack([columns[index] for index in members])
                rotation, _ = np.linalg.qr(rng.normal(size=(len(members), len(members))))
                mixed = block @ rotation
                for offset, index in enumerate(members):
                    columns[index] = mixed[:, offset]
    return np.column_stack(columns), np.asarray(values), np.asarray(points)


class _Problem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_levels_"))
        cls.problem = _translation(_silicon_input(cls.folder, spacing=1.2825, order=4)).problem
        cls.cell_grid = build_periodic_grid(cls.problem.periodic_cell, cls.problem.grid)
        cls.counts = tuple(int(value) for value in cls.cell_grid.points_per_axis)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.folder, ignore_errors=True)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: value}):
                self.assertEqual(levels.supercell_levels_setting(), expected)
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "dealt"}):
            with self.assertRaisesRegex(ValueError, levels.LEVELS_SWITCH):
                levels.supercell_levels_setting()

    def test_checked_before_the_scf(self) -> None:
        # the route's switches, read by both runners before the SCF (the levels are written after it)
        for value in ("on", "dealt"):
            with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: value}):
                with self.assertRaisesRegex(ValueError, levels.LEVELS_SWITCH):
                    route.validate_settings()
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "off"}):
            route.validate_settings()

    def test_the_note_of_the_route(self) -> None:
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "auto"}):
            self.assertIn("by their Bloch phases", route.describe("supercell", (2, 2, 2), per_cell=True))
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "off"}):
            text = route.describe("supercell", (2, 2, 2), per_cell=True)
        # off: the sentence of before
        self.assertNotIn("Bloch phases", text)
        self.assertTrue(text.endswith("parsec.dat and the result archive are the cell's"), text)

    def test_the_message_of_a_fixed_count_restart(self) -> None:
        from parsec_python.SCF.restart import RestartError, check_fixed_state_count

        check_fixed_state_count(21, 21, stored_kpoints=4)
        with self.assertRaises(RestartError) as caught:
            check_fixed_state_count(21, 24, stored_kpoints=4)
        self.assertIn("States_Num 21 or fewer", str(caught.exception))
        self.assertIn("PARSEC_KPOINT_SUPERCELL=off", str(caught.exception))
        # a file of one k-point (a cluster, the Gamma point): the message of before
        with self.assertRaises(RestartError) as caught:
            check_fixed_state_count(21, 24, stored_kpoints=1)
        self.assertNotIn("PARSEC_KPOINT_SUPERCELL", str(caught.exception))

    def test_translation_rows(self) -> None:
        rows = levels.translation_rows((2, 3, 1), (2, 1, 2))
        self.assertEqual(sorted(rows), [0, 2])
        field = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
        moved = field.reshape(-1)[rows[0]].reshape(4, 3, 2)
        np.testing.assert_array_equal(moved, np.roll(field, -2, axis=0))
        with self.assertRaises(ValueError):
            levels.translation_rows((2, 2, 2), (3, 1, 1))


class DenseStateTests(_Problem):
    def test_every_state_at_its_k_point(self) -> None:
        rng = np.random.default_rng(1)
        vectors, values, points = _bloch_states(self.counts, rng, 3)
        found, note = levels.state_mesh_points(
            SimpleNamespace(eigenvalues=values, wavefunctions=vectors), self.counts, (2, 2, 2)
        )
        self.assertIsNone(note)
        np.testing.assert_array_equal(found, points)

    def test_mixed_degenerate_levels_keep_their_k_points(self) -> None:
        rng = np.random.default_rng(2)
        vectors, values, points = _bloch_states(self.counts, rng, 4, clusters=(1, 3))
        found, note = levels.state_mesh_points(
            SimpleNamespace(eigenvalues=values, wavefunctions=vectors), self.counts, (2, 2, 2)
        )
        self.assertIsNone(note)
        for cluster in levels._clusters(values):
            self.assertEqual(sorted(found[cluster]), sorted(points[cluster]))

    def test_no_orbitals(self) -> None:
        found, note = levels.state_mesh_points(
            SimpleNamespace(eigenvalues=np.zeros(8), wavefunctions=None), self.counts, (2, 2, 2)
        )
        self.assertIsNone(found)
        self.assertIn("no orbitals", note)


class CellLevelTests(_Problem):
    def _result(self, per_point=6, clusters=(2,)):
        rng = np.random.default_rng(3)
        vectors, values, points = _bloch_states(self.counts, rng, per_point, clusters=clusters)
        occupations = np.where(values < 3.5, 1.0, 0.0)
        return SimpleNamespace(eigenvalues=values, occupations=occupations, wavefunctions=vectors), points

    def test_the_wedge_of_the_k_point_route(self) -> None:
        result, points = self._result()
        found = levels.cell_levels(result, self.problem, (2, 2, 2))
        self.assertIsNone(found.note)
        k_points, weights = kpoint_sampling(self.problem)
        np.testing.assert_allclose(found.k_points, k_points)
        np.testing.assert_allclose(found.weights, weights)
        self.assertEqual(found.per_k, 6)
        self.assertEqual(found.eigenvalues.size, 6 * len(weights))
        # each k-point of the wedge: the levels of its star, sorted, with their occupations
        sides = np.diag(np.asarray(self.problem.periodic_cell.lattice_vectors))
        for block, k_point in enumerate(k_points):
            bits = np.round(np.asarray(k_point) * sides / np.pi).astype(int) % 2
            point = int(np.ravel_multi_index(tuple(bits), (2, 2, 2)))
            expected = np.sort(result.eigenvalues[points == point])
            got = found.eigenvalues[6 * block : 6 * (block + 1)]
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(found.occupations[6 * block : 6 * (block + 1)], np.where(got < 3.5, 1.0, 0.0))
        # the union with the weights is the mesh's: the electrons of the supercell over its cells
        per_point = found.occupations.reshape(len(weights), -1).sum(axis=1)
        self.assertAlmostEqual(float(per_point @ found.weights), float(result.occupations.sum()) / 8.0)

    def test_the_files_of_the_cell(self) -> None:
        from parsec_python.Output.output_files import WavefunctionSelection
        from parsec_python.Output.parsec_dat import parsec_dat_from_result

        result, points = self._result()
        supercounts = tuple(2 * value for value in self.counts)
        density = np.tile(np.random.default_rng(4).random(self.counts), (2, 2, 2)).reshape(-1)
        result.next_effective_potential = -density
        result.density = density
        supercell = route._supercell_problem(self.problem, (2, 2, 2))
        self.assertEqual(tuple(route._grid_counts(supercell)), supercounts)
        stand_in, grid, problem = route.cell_parsec_dat_view(self.problem, (2, 2, 2))(result, None, None)
        self.assertEqual(stand_in.parsec_dat_notes, ())
        data, _ = parsec_dat_from_result(stand_in, grid, WavefunctionSelection(mode="list", indices=()),
                                         periodic_cell=problem.periodic_cell, monkhorst_pack_grid=(2, 2, 2))
        k_points, weights = kpoint_sampling(self.problem)
        self.assertEqual(len(data.levels[0]), len(weights))
        np.testing.assert_allclose(data.periodic.k_weights, weights)
        expected = levels.cell_levels(result, self.problem, (2, 2, 2))
        for block, level in enumerate(data.levels[0]):
            np.testing.assert_allclose(level.energies, expected.eigenvalues[6 * block : 6 * (block + 1)])
        # under off: the levels dealt in turn over the whole mesh, without a note (the output of before)
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "off"}):
            levels._CACHE.clear()
            stand_in, _, _ = route.cell_parsec_dat_view(self.problem, (2, 2, 2))(result, None, None)
        levels._CACHE.clear()
        self.assertEqual(stand_in.k_points.shape, (8, 3))
        self.assertEqual(stand_in.parsec_dat_notes, ())
        np.testing.assert_array_equal(stand_in.eigenvalues, result.eigenvalues[levels.dealt_levels(result.eigenvalues, 8)])
        np.testing.assert_allclose(stand_in.k_weights, np.full(8, 1.0 / 8.0))

    def test_unequal_counts_keep_the_fewest(self) -> None:
        result, points = self._result()
        # the highest level of the R point replaced by one more of Gamma (a supercell whose lowest 48 levels hold
        # seven of Gamma and five of R): every k-point keeps five, of the six of a k-point
        drop = np.flatnonzero(points == 7)[np.argmax(result.eigenvalues[points == 7])]
        keep = np.setdiff1d(np.arange(points.size), [drop])
        gamma = np.flatnonzero(points == 0)[0]
        cut = SimpleNamespace(
            eigenvalues=np.append(result.eigenvalues[keep], 100.0),
            occupations=np.append(result.occupations[keep], 0.0),
            wavefunctions=np.column_stack([result.wavefunctions[:, keep], result.wavefunctions[:, gamma]]),
        )
        levels._CACHE.clear()
        found = levels.cell_levels(cut, self.problem, (2, 2, 2))
        self.assertEqual(found.per_k, 5)
        self.assertEqual(found.states, 6)
        self.assertIn("5 to 7", found.note)
        self.assertIn("States_Num 5 or fewer", found.note)
        # the archive of the cell: five levels a k-point, the six states of a k-point of the run for --bands-from
        from parsec_python.cli import save_result_archive
        from parsec_python.models import EnergyBreakdown
        from parsec_python.SCF.bands import load_scf_for_bands

        rng = np.random.default_rng(6)
        fields = {name: np.tile(rng.random(self.counts), (2, 2, 2)).reshape(-1) for name in (
            "density", "core_density", "ionic_potential", "hartree_potential", "xc_potential",
            "input_effective_potential", "output_effective_potential", "next_effective_potential")}
        energies = EnergyBreakdown(eigenvalue=8.0, hartree=16.0, integral_vxc_rho=-8.0, exchange_correlation=-24.0,
                                   electron_ion=-80.0, ion_ion=-40.0, electronic=-100.0, total=-140.0)
        supercell = route._supercell_problem(self.problem, (2, 2, 2))
        full = SimpleNamespace(
            converged=True, iterations=3, atoms=tuple(supercell.atoms), electron_count=64.0,
            grid=build_periodic_grid(supercell.periodic_cell, supercell.grid), fermi_level=3.5, energies=energies,
            history=[], **vars(cut), **fields,
        )
        levels._CACHE.clear()
        path = self.folder / "parsec_python_results.npz"
        save_result_archive(path, route.cell_result_view(full, self.problem, (2, 2, 2)))
        with np.load(path) as archive:
            self.assertEqual(int(archive["n_states_per_kpoint"]), 5)
            self.assertEqual(int(archive["run_states_per_kpoint"]), 6)
            self.assertEqual(int(archive["n_kpoints"]), len(kpoint_sampling(self.problem)[1]))
        _, _, states = load_scf_for_bands(path, self.problem, SimpleNamespace(atoms=self.problem.atoms,
                                                                              grid=self.cell_grid), "scalar")
        self.assertEqual(states, 6)
        # sector labels of the supercell: none in the archive under auto; under off the archive of before (the
        # labels dealt with their levels, no run_states_per_kpoint)
        full.representations = (np.arange(cut.eigenvalues.size) % 4 + 1).astype(np.int32)
        levels._CACHE.clear()
        save_result_archive(path, route.cell_result_view(full, self.problem, (2, 2, 2)))
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["representations"], np.ones(5 * len(kpoint_sampling(self.problem)[1])))
        with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "off"}):
            levels._CACHE.clear()
            save_result_archive(path, route.cell_result_view(full, self.problem, (2, 2, 2)))
        levels._CACHE.clear()
        dealt = levels.dealt_levels(cut.eigenvalues, 8)
        with np.load(path) as archive:
            self.assertNotIn("run_states_per_kpoint", archive.files)
            self.assertEqual(int(archive["n_kpoints"]), 8)
            np.testing.assert_array_equal(archive["representations"], full.representations[dealt])
            np.testing.assert_array_equal(archive["eigenvalues_ry"], cut.eigenvalues[dealt])

    def test_more_levels_per_kpoint_fill_every_kpoint(self) -> None:
        # PARSEC_KPOINT_SUPERCELL_STATES=n: the supercell solves M (N + n) states and the cell's files keep N levels a
        # k-point; here N = 5 of the 6 Bloch states of every k-point, and the cut supercell of
        # test_unequal_counts_keep_the_fewest (5 to 7 levels a k-point) holds the 5 of every k-point
        result, points = self._result()
        drop = np.flatnonzero(points == 7)[np.argmax(result.eigenvalues[points == 7])]
        keep = np.setdiff1d(np.arange(points.size), [drop])
        gamma = np.flatnonzero(points == 0)[0]
        cut = SimpleNamespace(
            eigenvalues=np.append(result.eigenvalues[keep], 100.0),
            occupations=np.append(result.occupations[keep], 0.0),
            wavefunctions=np.column_stack([result.wavefunctions[:, keep], result.wavefunctions[:, gamma]]),
        )
        with mock.patch.dict(os.environ, {route.STATES_SWITCH: "1"}):
            levels._CACHE.clear()
            found = levels.cell_levels(cut, self.problem, (2, 2, 2))
            self.assertEqual(found.per_k, 5)
            self.assertEqual(found.states, 5)
            self.assertIsNone(found.note)
            for block in range(len(found.weights)):
                self.assertTrue(np.all(np.diff(found.eigenvalues[5 * block : 5 * (block + 1)]) >= 0.0))
            # the dealt levels (off): the cell's states of each k-point
            with mock.patch.dict(os.environ, {levels.LEVELS_SWITCH: "off"}):
                levels._CACHE.clear()
                dealt = levels.cell_levels(cut, self.problem, (2, 2, 2))
            self.assertEqual((dealt.per_k, dealt.states, dealt.eigenvalues.size), (5, 5, 40))
            # the supercell solves M (N + n) states
            supercell = route._supercell_problem(self.problem, (2, 2, 2), states=5)
            self.assertEqual(supercell.scf.number_of_states, 8 * 6)
        levels._CACHE.clear()
        self.assertEqual(route._supercell_problem(self.problem, (2, 2, 2), states=5).scf.number_of_states, 40)
        for value in ("-1", "some"):
            with mock.patch.dict(os.environ, {route.STATES_SWITCH: value}):
                with self.assertRaisesRegex(ValueError, route.STATES_SWITCH):
                    route.validate_settings()
        with mock.patch.dict(os.environ, {route.STATES_SWITCH: "cell"}):
            self.assertEqual(route.kpoint_supercell_extra_states(), 0)


class SectorCharacterTests(unittest.TestCase):
    def test_characters_of_a_group_with_the_translation(self) -> None:
        # a 4 x 1 x 1 supercell grid of a 2-point cell: the orbits {0, 2} and {1, 3} of the translation by one cell
        rows = levels.translation_rows((2, 1, 1), (2, 1, 1))
        orbitals = SimpleNamespace(
            full_to_wedge=np.array([0, 1, 0, 1]),
            phases=np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]),
        )
        np.testing.assert_array_equal(levels._character_bits(orbitals, rows), [0, 1])
        # a group without the translation: no characters
        orbitals.full_to_wedge = np.array([0, 1, 2, 3])
        self.assertIsNone(levels._character_bits(orbitals, rows))

    def test_partners_over_the_star(self) -> None:
        swap = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]])
        axes = [0, 1]
        self.assertEqual(sorted(levels._star(0b01, axes, swap)), [0b01, 0b10])
        self.assertEqual(levels._star(0b11, axes, swap), [0b11])
        # a weight-2 state at X_x: one partner at X_x, one at X_y; a weight-1 state keeps its own
        listed = levels._partner_bits([0b01, 0b11], [2, 1], [0.5, 0.5, 0.7], axes, swap)
        self.assertEqual(sorted(listed[:2].tolist()), [0b01, 0b10])
        self.assertEqual(int(listed[2]), 0b11)

    def test_weighted_sectors_without_the_translations(self) -> None:
        # a 4 x 4 x 1 supercell of a 2 x 2 x 1 cell: a solved state of weight 2 that mixes X_x and X_y (a sector
        # of a group without the translations), and one of Gamma of weight 1
        cell, repeats = (2, 2, 1), (2, 2, 1)
        rows = levels.translation_rows(cell, repeats)
        rng = np.random.default_rng(9)
        index = np.stack(np.meshgrid(np.arange(4) // 2, np.arange(4) // 2, [0], indexing="ij"), -1).reshape(-1, 3)

        def bloch(bits):
            periodic = np.tile(rng.normal(size=cell), (2, 2, 1)).reshape(-1)
            return periodic * np.where(index[:, :2] @ np.asarray(bits) % 2, -1.0, 1.0)

        solved = np.column_stack([bloch((1, 0)) + bloch((0, 1)), bloch((0, 0))])
        swap = np.array([np.eye(3, dtype=int), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]])
        with mock.patch.object(levels, "_sector_columns", lambda orbitals, columns: solved[:, np.asarray(columns)]):
            listed = levels._weighted_bits(None, np.array([0.5, 0.5, 0.9]), np.array([2, 1]), rows, [0, 1], swap, 8)
        self.assertEqual(sorted(listed[:2].tolist()), [0b01, 0b10])
        self.assertEqual(int(listed[2]), 0)


if __name__ == "__main__":
    unittest.main()
