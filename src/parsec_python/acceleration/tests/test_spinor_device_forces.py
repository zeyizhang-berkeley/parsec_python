"""Forces of the self-consistent spin-orbit loop of a cluster on the device (PARSEC_SOC_DEVICE_FORCES).

The contraction of :mod:`parsec_python.acceleration.Forces.spinor_device` runs here with NumPy too (the code the GPU
runs) against the readable :func:`parsec_python.Forces.spin_orbit.spinor_nonlocal_forces`, for whole-grid spinors and
for the sector spinors of the double group (each column with the electrons of its Kramers pair against the unfolded
pairs).  On a GPU the device forces of an accelerated spinor loop are compared with the readable forces of the same
result (whole grid and sectors), and the command line under ``auto`` with the reference solver under ``off``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Forces import build_nonlocal_projector_gradients, compute_spin_orbit_forces, compute_spinor_forces
from parsec_python.Forces.spin_orbit import build_spin_orbit_force_operator, spinor_nonlocal_forces
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.Forces.spinor_device import (
    SWITCH,
    SpinorForceContraction,
    contract_spinors,
    soc_device_forces_setting,
)
from parsec_python.acceleration.Hamiltonian.device_bloch import NumpyArrayBackend
from parsec_python.acceleration.SCF.soc_sectors import SectorRoute, SectorSpinors, detect_sector_plan
from parsec_python.acceleration.tests.test_soc_sectors import _AU, _H, _prepared
from parsec_python.tests.archives import read_archive

_CASES = {
    # the real Kramers sector of sigma_z with sigma_y
    "AuH on x": [Atom("Au", [1.4, 0.0, 0.0]), Atom("H", [-1.4, 0.0, 0.0])],
    # a complex sector (sigma_z alone)
    "AuH off the axis": [Atom("Au", [1.3, 0.1, 0.0]), Atom("H", [-1.5, 0.0, 0.0])],
    # a complex sector with the spin along y, inversion parity
    "Au2 in the xz plane": [Atom("Au", [0.6, 0.0, 0.4]), Atom("Au", [-0.6, 0.0, -0.4])],
}


def _operators(system, projectors):
    specifications = system.input.pseudopotentials
    gradients = build_nonlocal_projector_gradients(system.grid, system.atoms, system.pseudopotentials, specifications)
    spin_orbit = build_spin_orbit_force_operator(
        system.grid, system.atoms, system.pseudopotentials, specifications, projectors
    )
    return gradients, spin_orbit


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SWITCH, None)
            self.assertEqual(soc_device_forces_setting(), "auto")
        for value in ("auto", "off", " OFF "):
            with mock.patch.dict(os.environ, {SWITCH: value}):
                self.assertEqual(soc_device_forces_setting(), value.strip().lower())
        with mock.patch.dict(os.environ, {SWITCH: "on"}):
            with self.assertRaises(ValueError):
                soc_device_forces_setting()


class HostContractionTests(unittest.TestCase):
    """The contraction with NumPy against the readable spinor forces."""

    def test_whole_grid_spinors(self) -> None:
        rng = np.random.default_rng(5)
        for name, atoms in _CASES.items():
            with self.subTest(name):
                system, projectors = _prepared(atoms)
                gradients, spin_orbit = _operators(system, projectors)
                rows = 2 * system.grid.size
                spinors = rng.normal(size=(rows, 9)) + 1j * rng.normal(size=(rows, 9))
                spinors /= np.linalg.norm(spinors, axis=0)
                weights = np.linspace(1.0, 0.0, 9)
                expected = spinor_nonlocal_forces(
                    system.nonlocal_operator, gradients, spin_orbit, spinors, weights, len(atoms)
                )
                contraction = SpinorForceContraction(np, system.nonlocal_operator, gradients, spin_orbit)
                self.assertEqual(contract_spinors(contraction, spinors, weights), "device")
                found = contraction.forces(len(atoms))
                self.assertGreater(float(np.abs(expected).max()), 1.0e-3)
                np.testing.assert_allclose(found, expected, atol=1.0e-13 * float(np.abs(expected).max()))

    def test_the_terms_of_the_fortran_derived_routines(self) -> None:
        from parsec_python.Forces import spin_orbit_fortran
        from parsec_python.acceleration.Forces.spinor_device import fortran_spin_orbit_operator

        rng = np.random.default_rng(11)
        for name, atoms in _CASES.items():
            with self.subTest(name):
                system, projectors = _prepared(atoms)
                specifications = system.input.pseudopotentials
                gradients, _ = _operators(system, projectors)
                operator = fortran_spin_orbit_operator(
                    system.grid, system.atoms, system.pseudopotentials, specifications, projectors
                )
                rows = 2 * system.grid.size
                spinors = rng.normal(size=(rows, 7)) + 1j * rng.normal(size=(rows, 7))
                spinors /= np.linalg.norm(spinors, axis=0)
                weights = np.linspace(1.0, 0.1, 7)
                separable = spin_orbit_fortran.spinor_nonlocal_forces(
                    system.nonlocal_operator, gradients, spinors, weights, len(atoms)
                )
                spin_orbit = spin_orbit_fortran.spin_orbit_forces(
                    projectors,
                    spin_orbit_fortran.build_spin_orbit_projector_gradients(
                        system.grid, system.atoms, system.pseudopotentials, specifications
                    ),
                    spinors,
                    weights,
                    len(atoms),
                )
                contraction = SpinorForceContraction(np, system.nonlocal_operator, gradients, operator)
                contract_spinors(contraction, spinors, weights)
                found_separable, found_spin_orbit = contraction.parts(len(atoms))
                scale = max(float(np.abs(separable).max()), float(np.abs(spin_orbit).max()))
                np.testing.assert_allclose(found_separable, separable, atol=1.0e-13 * scale)
                np.testing.assert_allclose(found_spin_orbit, spin_orbit, atol=1.0e-13 * scale)
                self.assertGreater(float(np.abs(spin_orbit).max()), 1.0e-4)

    def test_sector_columns_stand_for_their_kramers_pairs(self) -> None:
        rng = np.random.default_rng(7)
        for name, atoms in _CASES.items():
            system, projectors = _prepared(atoms)
            gradients, spin_orbit = _operators(system, projectors)
            scalar = system.hamiltonian(system.ionic_potential)
            for parity in (False, True):
                plan, _ = detect_sector_plan(system.grid, system.atoms, parity=parity)
                with self.subTest(name, parity=parity, real=plan.real):
                    operator = SectorRoute(plan, NumpyArrayBackend(), projectors).hamiltonian(scalar, projectors)
                    sectors = getattr(operator, "sectors", [operator])
                    blocks = []
                    for sector in sectors:
                        values = rng.normal(size=(sector.shape[0], 5))
                        blocks.append(values if plan.real else values + 1j * rng.normal(size=values.shape))
                    # gaps, unused columns and the partner of the last pair beyond the count
                    places = (
                        [np.arange(5)]
                        if len(sectors) == 1
                        else [np.array([0, 2, 3, -1, 6]), np.array([1, -1, 4, 5, -1])]
                    )
                    count = 13 if len(sectors) == 2 else 9
                    states = SectorSpinors([s.parts for s in sectors], blocks, places, count)
                    occupations = np.repeat(np.linspace(1.0, 0.2, (count + 1) // 2), 2)[:count]
                    expected = spinor_nonlocal_forces(
                        system.nonlocal_operator,
                        gradients,
                        spin_orbit,
                        states.full_spinors(chunk=4),
                        occupations,
                        len(atoms),
                    )
                    contraction = SpinorForceContraction(np, system.nonlocal_operator, gradients, spin_orbit)
                    self.assertEqual(contract_spinors(contraction, states, occupations), "sectors")
                    found = contraction.forces(len(atoms))
                    np.testing.assert_allclose(found, expected, atol=1.0e-13 * float(np.abs(expected).max()))


class PlanCacheTests(unittest.TestCase):
    """The sector plan of a moved structure that keeps the operations, taken from the cache of the ionic loop."""

    def test_the_cached_plan_folds_the_same_numbers(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF import update_geometry
        from parsec_python.acceleration.SCF.soc_sectors import fold_operator

        system, projectors = _prepared(_CASES["AuH on x"])
        moved = update_geometry(
            system, [replace(atom, position=np.asarray(atom.position) + [0.05, 0.0, 0.0]) for atom in system.atoms]
        )
        moved_projectors = tuple(
            build_spin_orbit_projectors(moved.grid, moved.atoms, moved.pseudopotentials, moved.input.pseudopotentials)
        )
        cache: dict = {}
        first, _ = detect_sector_plan(system.grid, system.atoms, parity=False, cache=cache)
        second, _ = detect_sector_plan(moved.grid, moved.atoms, parity=False, cache=cache)
        fresh, _ = detect_sector_plan(moved.grid, moved.atoms, parity=False)
        self.assertIs(second, first)
        self.assertIsNot(fresh, first)
        laplacian = moved.negative_laplacian
        fold_operator(first, laplacian, system.nonlocal_operator, projectors)  # fills the static parts
        cached = fold_operator(second, laplacian, moved.nonlocal_operator, moved_projectors)
        made = fold_operator(fresh, laplacian, moved.nonlocal_operator, moved_projectors)
        for name in ("kinetic", "averages", "projectors", "soc", "unfold"):
            with self.subTest(name):
                a, b = getattr(cached, name), getattr(made, name)
                self.assertEqual((a != b).nnz, 0)
                np.testing.assert_array_equal(a.toarray(), b.toarray())
        # a structure that leaves the operations takes another plan
        broken = update_geometry(
            system, [replace(atom, position=np.asarray(atom.position) + [0.0, 0.07, 0.0]) for atom in system.atoms[:1]]
            + list(system.atoms[1:]),
        )
        other, _ = detect_sector_plan(broken.grid, broken.atoms, parity=False, cache=cache)
        self.assertIsNot(other, first)


class SphereTiesTests(unittest.TestCase):
    """PARSEC_SPHERE_TIES: the points exactly on the sphere of a grid through the origin, and the four-fold split."""

    _AU4 = [Atom("Au", [1.8, 1.8, 1.98]), Atom("Au", [-1.8, -1.8, 1.98]), Atom("Au", [1.8, -1.8, -1.98]),
            Atom("Au", [-1.8, 1.8, -1.98])]

    def _grids(self, shift):
        from parsec_python import GridSettings
        from parsec_python.Grid.cluster import build_cluster_grid
        from parsec_python.acceleration.Grid.cluster import build_cluster_grid_by_slabs

        settings = GridSettings(spacing=0.4, radius=10.0, expansion_order=8, shift=shift)
        return build_cluster_grid(settings), build_cluster_grid_by_slabs(settings)

    # The six orders of |(9, 12, 20)| h, on the sphere R = 25 h (h 0.4 bohr) of the grid through the origin, and whether
    # each is a point of PARSEC 1.5's grid of that sphere (gfortran -O2 for x86-64, read from its parsec.dat): the
    # points of the grid of PARSEC_SPHERE_TIES=fortran, all 65,179 of them, are those of that file.
    _PARSEC15 = {(9, 12, 20): True, (9, 20, 12): False, (12, 9, 20): True, (12, 20, 9): True, (20, 9, 12): False,
                 (20, 12, 9): True}

    def test_the_symmetric_rule_keeps_the_permutations_of_the_axes(self) -> None:
        from parsec_python.acceleration.SCF.soc_sectors import _four_fold_operation
        from parsec_python.Grid.cluster import SPHERE_TIES_SWITCH, sphere_ties_setting

        found, members = {}, {}
        for value in ("dot", "symmetric", "fortran", None):
            environment = {} if value is None else {SPHERE_TIES_SWITCH: value}
            with mock.patch.dict(os.environ, environment):
                if value is None:
                    os.environ.pop(SPHERE_TIES_SWITCH, None)
                    os.environ.pop("PARSEC_FORTRAN_NUMERICS", None)
                self.assertEqual(sphere_ties_setting(), value or "symmetric")
                for shift in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
                    readable, accelerated = self._grids(shift)
                    np.testing.assert_array_equal(readable.integer_coordinates, accelerated.integer_coordinates)
                    operation = _four_fold_operation(readable, self._AU4, 1.0e-6)
                    found[value, shift[0]] = (readable.size, None if operation is None else operation[1])
                    if shift[0] == 0.0:
                        rows = readable.rows_for_integer_coordinates(np.asarray(list(self._PARSEC15)))
                        members[value] = dict(zip(self._PARSEC15, (bool(row >= 0) for row in rows)))
        # the default is symmetric
        self.assertEqual(found[None, 0.0], found["symmetric", 0.0])
        # R = 25 h through the origin: the symmetric rule keeps all six orders and the four-fold images
        self.assertEqual(found["symmetric", 0.0], (65195, "S4z"))
        self.assertEqual(members["symmetric"], dict.fromkeys(self._PARSEC15, True))
        # (dot, the rule before, decides them by the order in which np.einsum sums the squares, which this test does
        # not pin: both builders take the same decisions, checked above)
        # PARSEC 1.5's grid: 16 points fewer; (x^2 + y^2) + z^2 keeps the swap of x and y, so S4z
        self.assertEqual(members["fortran"], self._PARSEC15)
        self.assertEqual(found["fortran", 0.0], (65179, "S4z"))
        # the shifted grid has no point on this sphere: the same grid under every value
        for value in ("dot", "fortran", None):
            self.assertEqual(found[value, 0.5], found["symmetric", 0.5])
        self.assertEqual(found["symmetric", 0.5][1], "S4z")
        with mock.patch.dict(os.environ, {SPHERE_TIES_SWITCH: "sorted"}):
            with self.assertRaises(ValueError):
                self._grids((0.0, 0.0, 0.0))

    def test_the_numerics_of_parsec_15_take_its_rule(self) -> None:
        from parsec_python.Grid.cluster import SPHERE_TIES_SWITCH, sphere_ties_setting

        with mock.patch.dict(os.environ, {"PARSEC_FORTRAN_NUMERICS": "on"}):
            os.environ.pop(SPHERE_TIES_SWITCH, None)
            self.assertEqual(sphere_ties_setting(), "fortran")
            readable, _ = self._grids((0.0, 0.0, 0.0))
            rows = readable.rows_for_integer_coordinates(np.asarray(list(self._PARSEC15)))
            self.assertEqual(dict(zip(self._PARSEC15, (bool(row >= 0) for row in rows))), self._PARSEC15)
            # a value set in the environment wins
            os.environ[SPHERE_TIES_SWITCH] = "symmetric"
            self.assertEqual(sphere_ties_setting(), "symmetric")


class RouteNoteTests(unittest.TestCase):
    """The route line of a structure is written where it differs from that of the structure before."""

    def test_a_structure_says_its_route_where_it_changes(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF import self_consistent_soc as module

        routes = iter([["sectors A"], ["sectors A"], ["sectors B"], ["NOTE: whole grid"], ["NOTE: whole grid"],
                       ["sectors A"]])

        def loop(system, projectors, *, note, **keywords):
            for line in next(routes):
                note(line)
            return SimpleNamespace()

        written: list[str] = []
        system = SimpleNamespace(grid=None, atoms=(), pseudopotentials={}, input=SimpleNamespace(pseudopotentials={}))
        with mock.patch.object(module, "run_self_consistent_soc", loop), mock.patch(
            "parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors", lambda *args, **keywords: []
        ):
            backend = module.soc_ionic_backend(note=written.append)
            for _ in range(6):
                backend.run_scf(system)
        self.assertEqual(written, ["sectors A", "sectors B", "NOTE: whole grid", "sectors A"])


class WarmStateTests(unittest.TestCase):
    """Which eigensolver state of the structure before a loop takes (the subspace of the same route only)."""

    def test_the_states_a_route_accepts(self) -> None:
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import SpinorEigvalState
        from parsec_python.acceleration.SCF.self_consistent_soc import _whole_grid_state
        from parsec_python.acceleration.SCF.soc_sectors import _SectorStates

        def state(rows, dtype):
            return SpinorEigvalState(
                vectors=np.zeros((rows, 4), dtype=dtype), eigenvalues=np.zeros(4), working_states=4,
                operator_dimension=rows, first_filter=False,
            )

        system, projectors = _prepared(_CASES["AuH on x"])
        scalar = system.hamiltonian(system.ionic_potential)
        plan, _ = detect_sector_plan(system.grid, system.atoms, parity=False)
        route = SectorRoute(plan, NumpyArrayBackend(), projectors)
        operator = route.hamiltonian(scalar, projectors)
        rows = int(operator.shape[0])
        self.assertTrue(plan.real)
        self.assertTrue(route.accepts(operator, state(rows, np.float64)))
        self.assertFalse(route.accepts(operator, state(rows, np.complex128)))  # a complex sector's vectors
        self.assertFalse(route.accepts(operator, state(2 * rows, np.complex128)))  # the whole grid's
        self.assertFalse(route.accepts(operator, _SectorStates([3], [state(rows, np.float64)])))
        self.assertTrue(_whole_grid_state(type("Op", (), {"shape": (2 * rows, 2 * rows)})(), state(2 * rows, complex)))
        self.assertFalse(_whole_grid_state(operator, _SectorStates([3], [None])))
        # the parity sectors of Au2: one state per sector of its dimension
        system, projectors = _prepared(_CASES["Au2 in the xz plane"], states=30)
        scalar = system.hamiltonian(system.ionic_potential)
        plan, _ = detect_sector_plan(system.grid, system.atoms, parity=True)
        route = SectorRoute(plan, NumpyArrayBackend(), projectors)
        operator = route.hamiltonian(scalar, projectors)
        sizes = [int(sector.shape[0]) for sector in operator.sectors]
        good = _SectorStates([5, 5], [state(size, np.complex128) for size in sizes])
        self.assertTrue(route.accepts(operator, good))
        self.assertTrue(route.accepts(operator, _SectorStates([5, 5], [None, state(sizes[1], np.complex128)])))
        self.assertFalse(route.accepts(operator, _SectorStates([5], [state(sizes[0], np.complex128)])))
        self.assertFalse(route.accepts(operator, state(sizes[0], np.complex128)))
        # a new loop filters the saved states again (solver%firstfilt)
        again = good.for_new_scf_loop()
        self.assertIsNot(again, good)
        self.assertEqual(again.counts, good.counts)
        self.assertTrue(all(sub.first_filter for sub in again.states))
        self.assertFalse(any(sub.first_filter for sub in good.states))


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class DeviceForceTests(unittest.TestCase):
    """The device forces of an accelerated spinor loop against the readable forces of the same result."""

    def test_device_forces_equal_the_readable_forces(self) -> None:
        from parsec_python.acceleration.Forces.spinor_device import compute_device_spinor_forces
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend
        from parsec_python.acceleration.SCF.self_consistent_soc import run_self_consistent_soc
        from parsec_python.acceleration.SCF.soc_sectors import SWITCH as SECTORS

        for name, states in (("AuH on x", 20), ("Au2 in the xz plane", 30)):
            system, projectors = _prepared(_CASES[name], spacing=0.6, radius=6.0, states=states, iterations=40)
            for route, expected_read in (("off", 2.0), ("kramers", 1.0), ("on", 1.0)):
                with self.subTest(name, route=route):
                    with mock.patch.dict(os.environ, {SECTORS: route}):
                        result = run_self_consistent_soc(
                            system, projectors, backend=CupyArrayBackend(0), keep_device=True
                        )
                    self.assertTrue(result.converged)
                    self.assertIsNotNone(result.device_spinors)
                    self.assertIsNotNone(result.eigensolver_state)
                    readable = compute_spinor_forces(system, result)
                    device = compute_device_spinor_forces(system, result)
                    self.assertIsNone(result.device_spinors)
                    self.assertEqual(device.phase_seconds["nonlocal_spinors_read"], expected_read)
                    scale = float(np.abs(readable.unsymmetrized).max())
                    self.assertGreater(scale, 1.0e-3)
                    for field in ("unsymmetrized", "total", "nonlocal_pseudopotential"):
                        np.testing.assert_allclose(
                            getattr(device, field), getattr(readable, field), atol=1.0e-12 * max(scale, 1.0)
                        )
                    np.testing.assert_allclose(device.local.total, readable.local.total, atol=1.0e-12)
                    # the host spinors of the result, uploaded a block at a time
                    uploaded = compute_device_spinor_forces(system, result)
                    self.assertEqual(uploaded.phase_seconds["nonlocal_spinors_read"], 3.0)
                    np.testing.assert_allclose(uploaded.total, readable.total, atol=1.0e-12 * max(scale, 1.0))
                    # averaged over the point group as the readable forces
                    symmetric = compute_device_spinor_forces(system, result, symmetrize=True)
                    np.testing.assert_allclose(
                        symmetric.total,
                        compute_spinor_forces(system, result, symmetrize=True).total,
                        atol=1.0e-12 * max(scale, 1.0),
                    )
                    # the Fortran-derived routines: their spin-orbit gradients, the term apart, the torque kept
                    fortran = compute_device_spinor_forces(system, result, symmetrize=True, fortran=True)
                    expected = compute_spin_orbit_forces(system, result, symmetrize=True)
                    for field in ("unsymmetrized", "total", "nonlocal_pseudopotential", "spin_orbit", "torque"):
                        np.testing.assert_allclose(
                            getattr(fortran, field), getattr(expected, field), atol=1.0e-12 * max(scale, 1.0)
                        )


_CLI_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.5
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 300.0
Max_Iter: 80
Convergence_Criterion: 1.0e-9

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

SO_from_scratch: .true.
{extra}
Atom_Types_Num: 2
Total_Atom_Num: 2

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   2.95 0 0
End atom_coord

Atom_Type: H
SO_PSP: .false.
Local_Component: s
Begin atom_coord
    0 0 0
End atom_coord

Correlation_Type: CA
"""


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class CommandLineTests(unittest.TestCase):
    """AuH through the accelerated command line: the device route (auto) against the reference solver (off)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _run(self, folder: str, extra: str, value: str, *arguments: str) -> tuple[int, str, dict]:
        from parsec_python.acceleration.cli import main

        directory = self.root / folder
        directory.mkdir()
        shutil.copy(_AU, directory / "Au_POTRE.DAT")
        shutil.copy(_H, directory / "H_POTRE.DAT")
        path = directory / "parsec.in"
        path.write_text(_CLI_INPUT.format(extra=extra), encoding="utf-8")
        with mock.patch.dict(os.environ, {SWITCH: value}):
            code = main([str(path), "--quiet", *arguments])
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        return code, log, read_archive(directory / "parsec_python_results.npz")

    def test_single_point_forces(self) -> None:
        device = self._run("auto", "Skip_Force: .false.\n", "auto", "--symmetry", "on")
        reference = self._run("off", "Skip_Force: .false.\n", "off")
        self.assertEqual(device[0], 0, device[1][-2000:])
        self.assertEqual(reference[0], 0, reference[1][-2000:])
        self.assertIn("Spin-orbit sectors", device[1])
        self.assertNotIn("running the reference parsec_python.cli solver", device[1])
        self.assertAlmostEqual(
            float(device[2]["energy_total_ry"]), float(reference[2]["energy_total_ry"]), delta=1.0e-8
        )
        np.testing.assert_allclose(device[2]["forces_total"], reference[2]["forces_total"], atol=1.0e-6)
        self.assertGreater(float(np.abs(reference[2]["forces_total"]).max()), 1.0e-3)

    def test_relaxation_continues_from_the_previous_structure(self) -> None:
        relax = "Skip_Force: .false.\nMinimization: bfgs\nMovement_Num: 2\nForce_Min: 0.0001\n"
        reference = self._run("off", relax, "off")
        # the whole grid (below the floor of the sectors) and the Kramers sector of --symmetry on, whose structures
        # filter the sector subspace of the one before
        for folder, arguments in (("auto", ()), ("sectors", ("--symmetry", "on"))):
            with self.subTest(folder):
                device = self._run(folder, relax, "auto", *arguments)
                self.assertIn(device[0], (0, 4), device[1][-2000:])
                self.assertEqual(device[0], reference[0])
                self.assertIn("the SCF loop continues from", device[1])
                self.assertEqual("Spin-orbit sectors" in device[1], bool(arguments))
                positions = device[2]["relaxation_positions"]
                self.assertEqual(positions.shape, reference[2]["relaxation_positions"].shape)
                np.testing.assert_allclose(positions, reference[2]["relaxation_positions"], atol=1.0e-5)
                np.testing.assert_allclose(
                    device[2]["relaxation_energies"], reference[2]["relaxation_energies"], atol=1.0e-7
                )
                # a structure after the first starts from the subspace of the one before: no first solve of its own
                # (the eigensolver detail of every structure lists the first solves of its loop)
                self.assertEqual(device[1].count("first_solves"), 1)
                self.assertEqual(device[1].count("subspace_steps"), len(positions))

    def test_a_fortran_relaxation_with_filter_restart(self) -> None:
        # the Fortran-derived routines (their forces, the archive of the general results) and Filter_Restart_After,
        # which drops the subspace of the structure before
        relax = (
            "Skip_Force: .false.\nMinimization: bfgs\nMovement_Num: 1\nForce_Min: 0.0001\nUse_Fortran_Routines: .true.\n"
            "Filter_Restart_After: 1\nRestart_Cheb_Remained_Times: 1\nRemained_Times_Atomic_Chrg_Den: 0\n"
        )
        device = self._run("auto", relax, "auto")
        reference = self._run("off", relax, "off")
        self.assertIn(device[0], (0, 4), device[1][-2000:])
        self.assertEqual(device[0], reference[0])
        self.assertEqual(device[1].count("Time for movement"), 2)
        self.assertIn("forces_spin_orbit", device[2])
        np.testing.assert_allclose(device[2]["forces_spin_orbit"], reference[2]["forces_spin_orbit"], atol=1.0e-6)
        np.testing.assert_allclose(
            device[2]["relaxation_positions"], reference[2]["relaxation_positions"], atol=1.0e-5
        )

    def test_a_relaxation_that_leaves_operations_says_its_new_sector(self) -> None:
        # bent H-Au-H (C2v about z in the xz plane) with one H fixed: the relaxation keeps sigma_y alone
        # (PARSEC_MOVE_FLAG_SUBGROUP=auto), so the second structure leaves the real sector of C2z for the complex
        # sector of sigma_y and parsec.out says so
        from parsec_python.acceleration.cli import main

        directory = self.root / "bent"
        directory.mkdir()
        shutil.copy(_AU, directory / "Au_POTRE.DAT")
        shutil.copy(_H, directory / "H_POTRE.DAT")
        text = _CLI_INPUT.format(
            extra="Skip_Force: .false.\nMinimization: bfgs\nMovement_Num: 1\nForce_Min: 0.0001\n"
        ).replace("Total_Atom_Num: 2", "Total_Atom_Num: 3")
        text = text.replace(
            "Local_Component: s\nBegin atom_coord\n   2.95 0 0\nEnd atom_coord",
            "Local_Component: s\nMove_Flag: -1\nBegin atom_coord\n   0 0 0.6 1\nEnd atom_coord",
        ).replace(
            "Local_Component: s\nBegin atom_coord\n    0 0 0\nEnd atom_coord",
            "Local_Component: s\nMove_Flag: -1\nBegin atom_coord\n   2.4 0 -1.3 0\n  -2.4 0 -1.3 1\nEnd atom_coord",
        )
        path = directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        code = main([str(path), "--quiet", "--symmetry", "on"])
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn(code, (0, 4), log[-2000:])
        lines = [line for line in log.splitlines() if line.startswith("Spin-orbit sectors")]
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("made real by T", lines[0])
        self.assertIn("sector +i of sigma_y:", lines[1])

    def test_molecular_dynamics(self) -> None:
        dynamics = (
            "Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 200\nT_Step: 10\nTime_Step: 20\nStep_Num: 2\nRng_Seed: 7\n"
        )
        device = self._run("auto", dynamics, "auto")
        reference = self._run("off", dynamics, "off")
        self.assertEqual(device[0], 0, device[1][-2000:])
        np.testing.assert_allclose(
            device[2]["dynamics_positions"], reference[2]["dynamics_positions"], atol=1.0e-5
        )
        np.testing.assert_allclose(device[2]["dynamics_energies"], reference[2]["dynamics_energies"], atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
