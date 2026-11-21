"""The group of a periodic cell about a defect off its origin (``PARSEC_PERIODIC_DEFECT_CENTRE``, 2026-11-19).

Before, the translation searches of the detection gave up after 32 candidates that passed their sampled atoms
(``_FRACTIONAL_BUDGET``): a vacancy at ``(a, a, a)`` of diamond cells of 3 x 3 x 3 cubes or more kept the 6 operations
of C3v of its 24 (at ``(a/2, a/2, 0)`` the 2 of a mirror), and so its sectors, the abelian sectors of a spin cell and
the little groups of the k-point sectors: the 999-atom V0 SCF took 46.8 s instead of 5.3 s (measured on 2026-11-19).
The searches that give up now take their candidates from the atoms of the rarest environment of the cell (at most 32,
every symmetry carries one of them onto another), and a cell whose pure translations repeat a defect takes its
operations about the centre of that environment where they are more.  These model tests (CPU) check the groups, that
the operations are those of the vacancy at the origin moved with it, that the weighted sectors give the spectrum of
the full grid, that the searches that do not give up and the crystals without a rare environment keep their
operations bit for bit, that noise in the positions below a loose atom tolerance does not hide the defect, the atom
tolerance of the k-point sectors (``PARSEC_SYMMETRY_ATOM_TOLERANCE``, ``PARSEC_KPOINT_SECTOR_TOLERANCE``) and that
both command lines refuse a bad value of these switches at the preparation; the SCF of a silicon vacancy moved by a
lattice vector (A100) its energy.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.Symmetry import axis_reflection as ar
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model, _weighted_spectrum
from parsec_python.acceleration.tests.test_periodic_symmetry_detection import _diamond
from parsec_python.models import Atom

_SWITCH = "PARSEC_PERIODIC_DEFECT_CENTRE"
# cubes of 6 bohr with 8 grid points per edge (steps of 0.75 bohr)
_SIDE = 6.0
_POINTS = 8


def _without(atoms, *sites):
    kept = [atom for atom in atoms if min(np.linalg.norm(np.asarray(atom.position) - site) for site in sites) > 1.0e-6]
    assert len(kept) == len(atoms) - len(sites)
    return kept


def _cell(repeats):
    atoms, sides = _diamond(repeats, side=_SIDE)
    grid, _ = _periodic_model(sides, _POINTS * repeats[0], 0.0, atoms, {"C": 2.0})
    return atoms, sides, grid


def _operations(grid, atoms, value: str):
    with mock.patch.dict(os.environ, {_SWITCH: value}):
        return ar._space_group_operations(grid, atoms, 1.0e-7, 1.0e-10)


def _detected(grid, atoms, value: str):
    with mock.patch.dict(os.environ, {_SWITCH: value}):
        return ar.detect_point_group_reduction(grid, atoms, ar.SignedPermutationReduction.detect(grid, atoms))


def _affine_set(matrices, shifts, counts) -> set:
    return {
        (np.asarray(matrix, dtype=np.int64).tobytes(), tuple(int(v) for v in np.mod(np.asarray(shift), counts)))
        for matrix, shift in zip(matrices, shifts)
    }


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {_SWITCH: value}):
                self.assertEqual(ar.periodic_defect_centre_setting(), expected)
        for value in ("on", "1"):
            with mock.patch.dict(os.environ, {_SWITCH: value}), self.assertRaises(ValueError):
                ar.periodic_defect_centre_setting()

    def test_small_queries_on_the_calling_thread(self) -> None:
        self.assertEqual(ar._PARALLEL_QUERY_POINTS, 16384)
        self.assertEqual(ar._query_workers(1), 1)
        self.assertEqual(ar._query_workers(ar._PARALLEL_QUERY_POINTS - 1), 1)
        self.assertEqual(ar._query_workers(ar._PARALLEL_QUERY_POINTS), ar._QUERY_WORKERS)


class VacancyTests(unittest.TestCase):
    """A vacancy of 3 x 3 x 3 cubes at the origin, at ``(a, a, a)`` and at ``(a/2, a/2, 0)``: the same structure
    moved by a lattice vector of the cube (the grid maps onto itself), so the same group, moved with it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.atoms, cls.sides, cls.grid = _cell((3, 3, 3))
        cls.counts = np.asarray(cls.grid.points_per_axis, dtype=np.int64)
        cls.sites = {"origin": np.zeros(3), "aaa": np.full(3, _SIDE), "half": np.array((3.0, 3.0, 0.0))}
        cls.cells = {name: _without(cls.atoms, site) for name, site in cls.sites.items()}

    def test_the_group_about_the_vacancy(self) -> None:
        reference = _operations(self.grid, self.cells["origin"], "auto")
        self.assertEqual(len(reference[0]), 24)
        self.assertTrue(all(not np.any(shift) for shift in reference[1]))
        for name in ("aaa", "half"):
            with self.subTest(site=name):
                before = _operations(self.grid, self.cells[name], "off")
                self.assertLess(len(before[0]), 24)
                matrices, shifts = _operations(self.grid, self.cells[name], "auto")
                self.assertEqual(len(matrices), 24)
                # the operations about the site s: r -> R r + (I - R) s, s in grid steps
                site = np.rint(self.sites[name] / (self.sides / self.counts)).astype(np.int64)
                moved = [site - np.asarray(matrix, dtype=np.int64) @ site for matrix in reference[0]]
                expected = _affine_set(reference[0], moved, self.counts)
                self.assertEqual(_affine_set(matrices, shifts, self.counts), expected)
                with mock.patch.dict(os.environ, {_SWITCH: "auto"}):
                    cell = sectors.CellGroup(self.grid, self.cells[name])
                    self.assertEqual(cell.order, 24)
                    self.assertIsNotNone(cell.table)
                    note = ar.defect_centre_note(self.grid, self.cells[name])
                self.assertIsNotNone(note)
                self.assertIn("4 C atoms about", note)
                self.assertIn(", ".join(f"{value:.4f}" for value in self.sites[name]), note)
                with mock.patch.dict(os.environ, {_SWITCH: "off"}):
                    self.assertIsNone(ar.defect_centre_note(self.grid, self.cells[name]))

    def test_the_sectors_of_every_route(self) -> None:
        # the weighted sectors of Td (subgroup of 8, weights 1 1 3 3) and the abelian group with translations of the
        # spin and hybrid cells (order 4), wherever the vacancy is
        for name, atoms in self.cells.items():
            with self.subTest(site=name):
                found = _detected(self.grid, atoms, "auto")
                self.assertIsInstance(found, ar.PointGroupReduction)
                self.assertEqual(found.full_operations.shape[0], 24)
                self.assertEqual(found.group_order, 8)
                np.testing.assert_array_equal(np.sort(found.sector_weights), [1, 1, 3, 3])
                with mock.patch.dict(os.environ, {_SWITCH: "auto"}):
                    abelian = ar.periodic_abelian_reduction(
                        self.grid, atoms, ar.SignedPermutationReduction.detect(self.grid, atoms)
                    )
                self.assertEqual(abelian.group_order, 4)
        self.assertEqual(_detected(self.grid, self.cells["aaa"], "off").group_order, 2)

    def test_the_seeded_search_is_the_exhaustive_one(self) -> None:
        # with every candidate matched in full (no budget) the search finds the translation the environment does
        atoms = self.cells["aaa"]
        species = ar._positions_by_species(atoms)
        period = ar._lattice_period(self.grid)
        steps = period / self.counts
        trees: dict = {}
        seed = ar._DefectSeed(species, period, 1.0e-7)
        self.assertIsNotNone(seed.environment())
        found = 0
        for operation in ar._signed_permutation_operations():
            if ar._integer_signed_permutation_parameters(self.grid, operation, 1.0e-10) is None:
                continue
            seeded = ar._translations_mapping(species, trees, operation, period, steps, self.counts, 1.0e-7, seed=seed)
            with mock.patch.object(ar, "_FRACTIONAL_BUDGET", 1 << 30):
                exhaustive = ar._translations_mapping(species, trees, operation, period, steps, self.counts, 1.0e-7)
            self.assertIsNot(seeded, ar._UNDECIDED)
            self.assertEqual(len(seeded), len(exhaustive))
            for left, right in zip(seeded, exhaustive):
                np.testing.assert_array_equal(left, right)
            found += bool(seeded)
        self.assertEqual(found, 24)

    def test_a_divacancy(self) -> None:
        # D3d about the bond centre, at the origin of a cube and moved by (a, a, a)
        for base in (np.zeros(3), np.full(3, _SIDE)):
            atoms = _without(self.atoms, base, base + 0.25 * _SIDE)
            with self.subTest(base=base.tolist()):
                self.assertLess(len(_operations(self.grid, atoms, "off")[0]), 12)
                self.assertEqual(len(_operations(self.grid, atoms, "auto")[0]), 12)


class WeightedSpectrumTests(unittest.TestCase):
    def test_a_vacancy_moved_by_a_lattice_vector(self) -> None:
        # 4 x 2 x 2 cubes (8,192 points): the vacancy at (a, a, a) keeps the 8 operations of D2d about it, whose
        # weighted sectors give the spectrum of the full grid and that of the vacancy at the origin
        import scipy.sparse.linalg as sla

        atoms, sides = _diamond((4, 2, 2), side=_SIDE)
        moved = _without(atoms, np.full(3, _SIDE))
        origin = _without(atoms, np.zeros(3))
        grid, hamiltonian = _periodic_model(sides, 4 * _POINTS, 0.0, moved, {"C": 2.0})
        _, origin_hamiltonian = _periodic_model(sides, 4 * _POINTS, 0.0, origin, {"C": 2.0})
        self.assertLess(len(_operations(grid, moved, "off")[0]), 8)
        found = _detected(grid, moved, "auto")
        self.assertIsInstance(found, ar.PointGroupReduction)
        self.assertEqual(found.full_operations.shape[0], 8)
        reference = _detected(grid, origin, "auto")
        self.assertEqual(reference.full_operations.shape[0], 8)
        count = 24
        # (dense spectra of every sector, a few more than the full grid gives)
        weighted = _weighted_spectrum(ReflectionRepresentationDecomposition.build(grid, found), hamiltonian, count + 8)
        full = np.sort(sla.eigsh(hamiltonian, k=count, sigma=-10.0, which="LM", tol=1.0e-12)[0])
        # Every eigenvalue of the full grid is one of the sectors', each copy matched with a copy of its own.  (ARPACK
        # can skip a copy of an exactly degenerate pair, a level of the E representation of D2d, and give the next
        # level in its place: on an A100 node once, -0.2062833952 found once of two; so not the other way round.)
        used = np.zeros(weighted.size, dtype=bool)
        for value in full:
            matches = np.flatnonzero(~used & (np.abs(weighted - value) <= 1.0e-9))
            self.assertGreater(matches.size, 0, f"{value} of the full grid is not among the sector spectra")
            used[matches[0]] = True
        self.assertGreaterEqual(int(np.sum(used[:count])), count - 2)
        at_origin = _weighted_spectrum(
            ReflectionRepresentationDecomposition.build(grid, reference), origin_hamiltonian, count + 8
        )
        np.testing.assert_allclose(weighted, at_origin, rtol=0, atol=1.0e-10)


class UnchangedTests(unittest.TestCase):
    """Cells whose searches do not give up, or without a rare environment: the same operations in the same order."""

    def test_the_same_operations(self) -> None:
        cases = []
        for repeats in ((2, 2, 2), (3, 3, 3), (2, 1, 1)):
            atoms, _, grid = _cell(repeats)
            cases.append((f"perfect {repeats}", grid, atoms))
        atoms, _, grid = _cell((3, 3, 3))
        cases.append(("vacancy at the origin", grid, _without(atoms, np.zeros(3))))
        atoms, _, grid = _cell((2, 2, 2))
        cases.append(("2x2x2 vacancy at (a, a, a)", grid, _without(atoms, np.full(3, _SIDE))))
        moved, sides = _diamond((2, 1, 1), side=_SIDE, offset=0.75 * np.array((1.0, 2.0, 3.0)))
        grid, _ = _periodic_model(sides, 2 * _POINTS, 0.0, moved, {"C": 2.0})
        cases.append(("2x1x1 off the origin", grid, moved))
        for label, grid, atoms in cases:
            with self.subTest(label):
                before, after = _operations(grid, atoms, "off"), _operations(grid, atoms, "auto")
                self.assertEqual(len(before[0]), len(after[0]))
                for part in (0, 1):
                    for left, right in zip(before[part], after[part]):
                        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                off, auto = _detected(grid, atoms, "off"), _detected(grid, atoms, "auto")
                self.assertIs(type(off), type(auto))
                np.testing.assert_array_equal(off.full_to_wedge, auto.full_to_wedge)
                if isinstance(off, ar.PointGroupReduction):
                    np.testing.assert_array_equal(off.full_operations, auto.full_operations)
                    np.testing.assert_array_equal(off.full_translations, auto.full_translations)
                with mock.patch.dict(os.environ, {_SWITCH: "auto"}):
                    self.assertIsNone(ar.defect_centre_note(grid, atoms))

    def test_a_repeated_defect(self) -> None:
        # three vacancies a third of the cell apart: a pure translation that is not a half-cell one, so the
        # operations are about a centre; that of the rarest environment, an empty site, keeps those of D2d
        atoms, sides = _diamond((6, 3, 3), side=_SIDE)
        grid, _ = _periodic_model(sides, 6 * _POINTS, 0.0, atoms, {"C": 2.0})
        repeated = _without(atoms, *[np.array((_SIDE + 2.0 * _SIDE * index, _SIDE, _SIDE)) for index in range(3)])
        self.assertEqual(len(_operations(grid, repeated, "off")[0]), 2)
        self.assertEqual(len(_operations(grid, repeated, "auto")[0]), 8)
        with mock.patch.dict(os.environ, {_SWITCH: "auto"}):
            self.assertEqual(sectors.CellGroup(grid, repeated).order, 8)

    def test_the_spin_guard_keeps_its_environments(self) -> None:
        # the guard of the spin sectors classes the environments with the shared code: Td about the vacancy
        from parsec_python.acceleration.SCF.spin_sectors import _environment_operations

        atoms, _, grid = _cell((3, 3, 3))
        found = _environment_operations(grid, _without(atoms, np.full(3, _SIDE)), 1.0e-7)
        self.assertEqual(len(found), 23)


def _noisy(atoms, seed: int, size: float = 1.0e-5):
    rng = np.random.default_rng(seed)
    return [Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.uniform(-1.0, 1.0, 3) * size)) for atom in atoms]


class NoisyPositionTests(unittest.TestCase):
    """Atoms off their sites by up to 1e-5 bohr per axis at ``PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-4`` (a structure from
    another code): the rarest environment is chosen within the tolerance, not among classes rounded to 1e-4 bohr
    (which the noise splits into classes of one bulk atom), so the vacancy keeps Td wherever it is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.atoms, cls.sides, cls.grid = _cell((3, 3, 3))
        cls.period = ar._lattice_period(cls.grid)

    def _environments(self, atoms):
        species = ar._positions_by_species(atoms)
        trees = {name: ar._matching_tree(species[name], self.period) for name in species}
        return species, ar._atom_environments(species, self.period, trees)

    def test_a_noisy_vacancy_off_the_origin(self) -> None:
        # seeds whose noise hides the vacancy from the rounded classes (the choice of before, kept by the spin guard)
        window = 4.0e-4 + 1.0e-9
        for name, site, seeds in (("aaa", np.full(3, _SIDE), (1, 2, 7)), ("half", np.array((3.0, 3.0, 0.0)), (0, 2, 4))):
            for seed in seeds:
                atoms = _noisy(_without(self.atoms, site), seed)
                with self.subTest(site=name, seed=seed):
                    species, environments = self._environments(atoms)
                    self.assertGreater(int(np.sum(ar._rarest_class(environments)[2] <= window)), 32)
                    environment = ar._rarest_environment(species, self.period, 1.0e-4)
                    self.assertIsNotNone(environment)
                    self.assertEqual(environment.alike.size, 4)
                    # the vacancy, modulo the lattice
                    offset = environment.centre(species, self.period) - site
                    offset -= self.period * np.rint(offset / self.period)
                    self.assertLess(float(np.max(np.abs(offset))), 1.0e-4)
                    self.assertEqual(len(_operations_at(self.grid, atoms, "auto", 1.0e-4)[0]), 24)
                    self.assertLess(len(_operations_at(self.grid, atoms, "off", 1.0e-4)[0]), 24)

    def test_the_choice_of_before_in_exact_cells(self) -> None:
        # where the rounded classes are those of the tolerance the same atom is chosen: vacancies, a divacancy, a
        # repeated defect and perfect cells
        cases = [_without(self.atoms, site) for site in (np.zeros(3), np.full(3, _SIDE), np.array((3.0, 3.0, 0.0)))]
        cases.append(_without(self.atoms, np.full(3, _SIDE), np.full(3, 1.25 * _SIDE)))
        cases.append(self.atoms)
        for index, atoms in enumerate(cases):
            with self.subTest(case=index):
                _, environments = self._environments(atoms)
                before, now = ar._rarest_class(environments), ar._rarest_class(environments, 1.0e-7)
                self.assertEqual(before[:2], now[:2])
                np.testing.assert_array_equal(before[2], now[2])
        repeated, sides = _diamond((6, 3, 3), side=_SIDE)
        repeated = _without(repeated, *[np.array((_SIDE + 2.0 * _SIDE * index, _SIDE, _SIDE)) for index in range(3)])
        species = ar._positions_by_species(repeated)
        period = np.asarray(sides, dtype=np.float64)
        trees = {name: ar._matching_tree(species[name], period) for name in species}
        environments = ar._atom_environments(species, period, trees)
        self.assertEqual(ar._rarest_class(environments)[:2], ar._rarest_class(environments, 1.0e-7)[:2])

    def test_the_counts_of_alike_environments(self) -> None:
        # a crystal class of 300 atoms (100 of them exact copies, 200 within 4e-5), a defect class of 4 within 3e-5,
        # one of 6 at 3e-4 from it (within four tolerances), single atoms and an empty list
        rng = np.random.default_rng(11)
        base = rng.uniform(2.0, 5.0, 12)
        rows = [np.repeat(base[None, :], 100, axis=0), base + rng.uniform(-2.0e-5, 2.0e-5, (200, 12))]
        defect = base + 0.1
        rows.append(defect + rng.uniform(-1.5e-5, 1.5e-5, (4, 12)))
        rows.append(defect + 3.0e-4 + rng.uniform(-1.0e-6, 1.0e-6, (6, 12)))
        rows.append(base[None, :] - 0.2 - 0.05 * np.arange(5)[:, None])
        rows = np.vstack(rows)
        window = 4.0e-4 + 1.0e-9
        counts = ar._alike_counts(rows, window, 32)
        brute = np.array([np.sum(np.max(np.abs(rows - row[None, :]), axis=1) <= window) for row in rows])
        loose = np.array([np.sum(np.max(np.abs(rows - row[None, :]), axis=1) <= 1.5 * window) for row in rows])
        small = brute <= 32
        self.assertTrue(np.all(counts[~small] > 32))
        self.assertTrue(np.all(counts[small] >= brute[small]))
        self.assertTrue(np.all(counts[small] <= loose[small]))
        np.testing.assert_array_equal(counts[300:], [10] * 10 + [1] * 5)
        self.assertEqual(ar._alike_counts(np.zeros((0, 12)), window, 32).size, 0)
        np.testing.assert_array_equal(ar._alike_counts(np.zeros((3, 0)), window, 32), [3, 3, 3])


def _operations_at(grid, atoms, value: str, tolerance: float):
    with mock.patch.dict(os.environ, {_SWITCH: value}):
        return ar._space_group_operations(grid, atoms, tolerance, 1.0e-10)


class KPointToleranceTests(unittest.TestCase):
    """The k-point sectors honour ``PARSEC_SYMMETRY_ATOM_TOLERANCE``, at most 1e-6 bohr with forces; 1e-7 under
    ``PARSEC_KPOINT_SECTOR_TOLERANCE=fixed``."""

    def test_the_tolerance(self) -> None:
        for value, plain, forces in (("", 1.0e-7, 1.0e-7), ("1e-4", 1.0e-4, 1.0e-6), ("1e-8", 1.0e-8, 1.0e-8)):
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": value}):
                self.assertEqual(sectors.kpoint_sector_atom_tolerance(), plain)
                self.assertEqual(sectors.kpoint_sector_atom_tolerance(forces=True), forces)
                for setting in ("", "follow", " Follow "):
                    with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_TOLERANCE": setting}):
                        self.assertEqual(sectors.kpoint_sector_tolerance_setting(), "follow")
                        self.assertEqual(sectors.kpoint_sector_atom_tolerance(), plain)
                with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_TOLERANCE": "FIXED"}):
                    self.assertEqual(sectors.kpoint_sector_tolerance_setting(), "fixed")
                    self.assertEqual(sectors.kpoint_sector_atom_tolerance(), 1.0e-7)
                    self.assertEqual(sectors.kpoint_sector_atom_tolerance(forces=True), 1.0e-7)
        for setting in ("off", "auto", "1e-4"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_TOLERANCE": setting}):
                with self.assertRaises(ValueError):
                    sectors.kpoint_sector_tolerance_setting()
                with self.assertRaises(ValueError):
                    sectors.check_kpoint_sector_switches()
        # (a bad atom tolerance is refused under fixed as well)
        with mock.patch.dict(
            os.environ, {"PARSEC_KPOINT_SECTOR_TOLERANCE": "fixed", "PARSEC_SYMMETRY_ATOM_TOLERANCE": "abc"}
        ), self.assertRaises(ValueError):
            sectors.kpoint_sector_atom_tolerance()

    def test_the_switches_read_at_the_preparation(self) -> None:
        sectors.check_kpoint_sector_switches()
        for name, value in (
            ("PARSEC_KPOINT_SECTORS", "on"),
            ("PARSEC_SYMMETRY_ATOM_TOLERANCE", "abc"),
            ("PARSEC_PERIODIC_TRANSLATIONS", "all"),
            ("PARSEC_PERIODIC_TRIVIAL_ORIGIN", "on"),
            ("PARSEC_PERIODIC_SYMMETRY_CENTRE", "on"),
            (_SWITCH, "bogus"),
        ):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, name):
                    sectors.check_kpoint_sector_switches()

    def _geometry(self, atoms, grid, sides, cache=None):
        return {
            "grid": grid,
            "atoms": tuple(atoms),
            "k_points": np.zeros((1, 3)),
            "weights": np.ones(1),
            "lattice": np.diag(sides),
            "density_symmetry": None,
            "cache": cache,
        }

    def test_the_plans_take_the_tolerance_and_the_cache_first(self) -> None:
        from parsec_python.SCF.kpoint_solver import KPointJob

        atoms, sides = _diamond((1, 1, 1), side=_SIDE)
        rng = np.random.default_rng(3)
        noisy = [Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.normal(0.0, 1.0e-5, 3))) for atom in atoms]
        grid, _ = _periodic_model(sides, _POINTS, 0.0, noisy, {"C": 2.0})
        cache: dict = {}
        # (40 states of the 512 points: the 16 sectors of the Gamma point halve the work)
        job = KPointJob(1, 40, None, None, None, geometry=self._geometry(noisy, grid, sides, cache))
        forces = KPointJob(1, 40, None, None, None, geometry=self._geometry(noisy, grid, sides), extras=lambda *a: 0.0)
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0):
            # the noisy atoms keep no operation at 1e-7
            self.assertIsNone(sectors.plan_sectors(job))
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4"}):
                plans = sectors.plan_sectors(job)
                self.assertIsNotNone(plans)
                self.assertEqual(plans[0].guard_group.cell.atom_tolerance, 1.0e-4)
                # the next loop of the structure takes the plans before any search of the group
                with mock.patch.object(sectors, "CellGroup", side_effect=AssertionError("searched again")):
                    self.assertIs(sectors.plan_sectors(job)[0], plans[0])
                # a loop with forces: the group of 1e-6 bohr, which the atoms 1e-5 off do not keep
                self.assertIsNone(sectors.plan_sectors(forces))
                # PARSEC_KPOINT_SECTOR_TOLERANCE=fixed: 1e-7 as before (another signature: no plans from the cache)
                with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_TOLERANCE": "fixed"}):
                    self.assertIsNone(sectors.plan_sectors(job))

    def test_the_guard_tests_the_operations_the_atoms_keep(self) -> None:
        # a cell group at a loose tolerance: the guard tests only the operations that hold the atoms to 1e-6 bohr
        atoms, sides = _diamond((1, 1, 1), side=_SIDE)
        rng = np.random.default_rng(5)
        noisy = [Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.normal(0.0, 1.0e-5, 3))) for atom in atoms]
        grid, _ = _periodic_model(sides, _POINTS, 0.0, noisy, {"C": 2.0})

        class Everything:
            def __call__(self, index):
                return True

            def in_density_group(self, index):
                return False

        exact = sectors.SectorGuardGroup(sectors.CellGroup(grid, atoms), Everything())
        self.assertIsNone(exact._strict_test())
        self.assertGreater(len(exact.images()), 0)
        loose_cell = sectors.CellGroup(grid, noisy, atom_tolerance=1.0e-4)
        self.assertEqual(loose_cell.order, exact.cell.order)
        loose = sectors.SectorGuardGroup(loose_cell, Everything())
        self.assertIsNotNone(loose._strict_test())
        self.assertEqual(loose.images(), [])


# H2 in a bulk cell with a 2 x 2 x 2 mesh: a k-point run, which the accelerated command line hands to the reference one
_KPOINT_INPUT = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   7.2  7.2  7.2
end Cell_Shape
Grid_Spacing: 0.6
States_Num: 4
Correlation_Type: pbe
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
   2 2 2
end Monkhorst_Pack_Grid
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.0 3.1 1.8
  3.0 3.1 3.2
end Atom_Coord
"""


class PreparationRefusalTests(unittest.TestCase):
    """A value the switches of the k-point sectors and of their detection do not know is refused by either command
    line at the preparation of a periodic run, before any output (it was refused only at the setup of the k-point
    solver of the reference one, after its preparation)."""

    def test_both_command_lines(self) -> None:
        import shutil
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        from pathlib import Path

        from parsec_python.acceleration.cli import main as accelerated_main
        from parsec_python.cli import main as reference_main

        data = Path(__file__).resolve().parents[2] / "tests" / "data"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            for label, main, options in (
                ("accelerated", accelerated_main, ["--device", "cpu"]),
                ("reference", reference_main, []),
            ):
                for name, value in (
                    (_SWITCH, "bogus"),
                    ("PARSEC_SYMMETRY_ATOM_TOLERANCE", "abc"),
                    ("PARSEC_PERIODIC_SYMMETRY_CENTRE", "on"),
                    ("PARSEC_KPOINT_SECTOR_TOLERANCE", "loose"),
                ):
                    folder = Path(root) / f"{label}_{name}"
                    folder.mkdir()
                    shutil.copy(data / "spin_pbe_H_POTRE.DAT", folder / "H_POTRE.DAT")
                    path = folder / "parsec.in"
                    path.write_text(_KPOINT_INPUT, encoding="utf-8")
                    errors = StringIO()
                    with self.subTest(cli=label, name=name), mock.patch.dict(os.environ, {name: value}):
                        with redirect_stdout(StringIO()), redirect_stderr(errors):
                            code = main([str(path), "--quiet", *options])
                        self.assertEqual(code, 2, errors.getvalue())
                        self.assertIn(f"Input error: {name}", errors.getvalue())
                        self.assertEqual(sorted(item.name for item in folder.iterdir()), ["H_POTRE.DAT", "parsec.in"])


def _vacancy_problem(offset_steps=None):
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon
    from parsec_python.acceleration.tests.test_periodic_symmetry_detection import _moved, _repeated

    base = _repeated(silicon("pbe", spacing=0.64), (4, 2, 2))
    atoms = tuple(atom for atom in base.atoms if np.linalg.norm(np.asarray(atom.position)) > 1.0e-6)
    assert len(atoms) == len(base.atoms) - 1
    problem = replace(base, atoms=atoms)
    return problem if offset_steps is None else _moved(problem, offset_steps)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class VacancySCFTests(unittest.TestCase):
    def _run(self, problem, environment):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
            forces = driver.compute_forces(system, result)
        return result, dict(result.backend.details), np.asarray(forces.total)

    def test_a_silicon_vacancy_moved_by_a_lattice_vector(self) -> None:
        # Si 4 x 2 x 2 cubes, vacancy at the origin and moved by (a, a, a) (16 grid steps per cube edge): the same
        # structure on the same grid, the same group (D2d, 8 operations), the same energy
        from parsec_python.Grid.pbc import build_periodic_grid

        origin_problem = _vacancy_problem()
        grid = build_periodic_grid(origin_problem.periodic_cell, origin_problem.grid)
        per_cube = int(grid.points_per_axis[1]) // 2
        moved_problem = _vacancy_problem((per_cube, per_cube, per_cube))
        origin, origin_details, origin_forces = self._run(origin_problem, {})
        moved, moved_details, moved_forces = self._run(moved_problem, {})
        before, before_details, before_forces = self._run(moved_problem, {_SWITCH: "off"})
        # (about the origin the 8 operations have no translation)
        self.assertIn("exact point group of 8 signed permutations", origin_details["symmetry_detection"])
        self.assertIn("exact group of 8 operations", moved_details["symmetry_detection"])
        self.assertIn(_SWITCH, moved_details["symmetry_detection"])
        self.assertNotIn(_SWITCH, origin_details["symmetry_detection"])
        self.assertNotIn("of 8 ", before_details.get("symmetry_detection", ""))
        for result in (origin, moved, before):
            self.assertTrue(result.converged)
        for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
            self.assertAlmostEqual(
                getattr(moved.energies, field), getattr(origin.energies, field), delta=2.0e-6, msg=field
            )
            self.assertAlmostEqual(
                getattr(before.energies, field), getattr(origin.energies, field), delta=2.0e-6, msg=field
            )
        # the forces of the neighbours of the vacancy, in the sectors of the group about it (the atoms in the same
        # order: the moved problem moves every atom)
        self.assertGreater(float(np.abs(origin_forces).max()), 1.0e-3)
        np.testing.assert_allclose(moved_forces, origin_forces, rtol=0, atol=1.0e-5)
        np.testing.assert_allclose(before_forces, origin_forces, rtol=0, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
