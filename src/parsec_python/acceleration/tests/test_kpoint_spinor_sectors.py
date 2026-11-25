"""The double-group sectors of the k-points of a periodic spin-orbit SCF (``PARSEC_SOC_KPOINT_SECTORS``), on the host.

The device code runs with NumPy as the array backend (``PARSEC_KPOINT_BACKEND=numpy-device``).  Checked: the union
of the spectra of the sectors and their Kramers partners against the dense spinor Bloch Hamiltonian of the whole grid
(an atom with the whole cubic group, a pair along z, a cell with only a mirror normal to y: the frame of ``|+-y>``);
their unfolded eigenvectors and partners as eigenvectors of the whole operator; the SCF with forces against the
loop without sectors on a set of k-points the cell group maps onto itself; a collinear magnetization along z
(``PARSEC_SOC_SPIN_KPOINT_SECTORS=on``: the sectors about z without pairs, operator and SCF); the Kramers partners
through an operation ``h`` whose spin turn is ``-i sigma_a`` (``C2a`` or ``sigma_a``, forced where the inversion would be
taken first: a pair of atoms with the group ``{E, i, C2x, sigma_x}``, the frame of ``|+-x>``), the partner density as the
image under ``h``, and SCF and forces against the whole grid with them; the states handed from one solver to the next
(``export_states``, ``import_states``); the switch values and the floor.
"""

from __future__ import annotations

import functools
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
import parsec_python.acceleration  # noqa: F401  (installs the parallel k-point backend)
from parsec_python.SCF.kpoint_solver import KPointJob
from parsec_python.SCF.kpoint_weights import validated_kpoint_weights
from parsec_python.SCF.kpoints_soc import (
    _k_point_operators,
    _sector_geometry,
    _spinor_hamiltonian,
    run_self_consistent_soc_kpoints,
    run_self_consistent_soc_kpoints_spin_polarized,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
from parsec_python.acceleration.SCF.kpoint_spinor_sectors import (
    SPIN_SWITCH,
    SWITCH,
    SpinorSectorKPointSolver,
    plan_spinor_sectors,
    soc_kpoint_sectors_setting,
)

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_BOX = 8.0
_STEP = _BOX / 9.0
_CASES = {
    "atom": [Atom("Au", [_BOX / 2] * 3)],
    "pair": [Atom("Au", [_BOX / 2, _BOX / 2, _BOX / 2 - 1.6]), Atom("Au", [_BOX / 2, _BOX / 2, _BOX / 2 + 1.6])],
    "mirror": [Atom("Au", [_BOX / 2] * 3), Atom("Au", [_BOX / 2 + 2 * _STEP, _BOX / 2, _BOX / 2 + 1.0])],
    # {E, i, C2x, sigma_x} about the centre of the cell (C2h about x): sectors about x
    "c2hx": [Atom("Au", [_BOX / 2, _BOX / 2 + 1.3, _BOX / 2 + 0.8]), Atom("Au", [_BOX / 2, _BOX / 2 - 1.3, _BOX / 2 - 0.8])],
}


def _turned_partners():
    """``_antiunitary`` replaced by the first candidate whose spin turn is ``-i sigma_a`` (``C2a`` or ``sigma_a``), the
    inversion only where there is none: the path of those partners, which the order of the cell group leaves to the
    inversion wherever both exist."""

    from parsec_python.acceleration.SCF import kpoint_spinor_sectors as module

    def choose(cell, k_point, elements, axis, invariance):
        found = module.antiunitary_candidates(cell, k_point, elements, axis, invariance)
        turned = [entry for entry in found if entry[1] != module._IDENTITY_TYPE]
        chosen = turned or found
        return chosen[0] if chosen else (None, None)

    return mock.patch.object(module, "_antiunitary", choose)
# Gamma and the star of (0, 0, 1/3): a set every operation of the cubic group maps onto itself
_FRACTIONS = np.array(
    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
) / 3.0
_DEVICE = {"PARSEC_KPOINT_BACKEND": "numpy-device"}


@functools.lru_cache(maxsize=None)
def _system(case: str, symmetrized: bool = True, criterion: float = 1.0e-11, magnetic: bool = False):
    """The cell of ``case``; ``symmetrized``: the density averaged over its space group (``Symmetrize_Charge_Density``),
    so that the operations of the Kramers pairs keep the potential at every step; ``magnetic``: spin-polarized, an
    initial polarization of 0.3."""
    from parsec_python.Grid.space_group import find_space_group

    atoms = _CASES[case]
    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True, initial_spin_polarization=0.3 if magnetic else 0.0)
        },
        grid=PeriodicGridSettings(spacing=0.89, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=_BOX * np.eye(3)),
        scf=SCFSettings(
            max_iterations=60,
            number_of_states=20 * len(atoms),
            convergence_criterion=criterion,
            self_consistent_spin_orbit=True,
            fermi_temperature_kelvin=100.0 if magnetic else 1000.0,
            symmetrize_density=symmetrized,
            spin_polarized=magnetic,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=criterion / 10),
        space_group=find_space_group(_BOX * np.eye(3), atoms) if symmetrized else None,
    )
    return prepare_periodic_single_point(problem)


def _k_points():
    k_points = 2.0 * np.pi / _BOX * _FRACTIONS
    weights = np.full(len(k_points), 1.0 / len(k_points))
    return validated_kpoint_weights(k_points, weights, "test")


def _plans(system, k_points, weighting, states: int, magnetic: bool = False):
    geometry = _sector_geometry(system, k_points, weighting, False, magnetic=magnetic)
    job = KPointJob(len(k_points), states, None, None, None, geometry=geometry)
    notes: list = []
    return plan_spinor_sectors(job, notes), notes


class SpinorSectorOperatorTests(unittest.TestCase):
    def check(self, case: str, axes: set, symmetrized: bool = True, magnetic: bool = False) -> None:
        system = _system(case, symmetrized)
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {SWITCH: "on", SPIN_SWITCH: "on" if magnetic else "off"}):
            plans, notes = _plans(system, k_points, weighting, int(system.input.scf.number_of_states), magnetic)
        # a field of the symmetry of the cell for the collinear term
        delta = 0.02 * system.ionic_potential / np.max(np.abs(system.ionic_potential)) if magnetic else None
        self.assertIsNotNone(plans, notes)
        converter = DeviceBlochConverter(NumpyArrayBackend())
        operators = _k_point_operators(system, k_points)
        split = 0
        for index, k_point in enumerate(k_points):
            plan = plans[index]
            if plan is None:
                continue
            split += 1
            self.assertIn(plan.axis, axes)
            nonlocal_k, soc_k = operators[index]
            full = _spinor_hamiltonian(system, k_point, nonlocal_k, soc_k, system.ionic_potential, xc_delta=delta)
            if magnetic:
                self.assertIsNone(plan.antiunitary)
            dense = full.apply(np.eye(full.shape[0], dtype=complex))
            whole = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
            values = []
            for sector in plan.solved:
                operator = plan.hamiltonian(sector, full).device_operator(converter)
                matrix = operator.apply(np.eye(operator.shape[0], dtype=complex))
                self.assertLess(np.max(np.abs(matrix - matrix.conj().T)), 1.0e-12)
                levels, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
                values.append(levels)
                spinors = plan.unfold(sector, vectors[:, :6])
                np.testing.assert_allclose(np.linalg.norm(spinors, axis=0), 1.0, atol=1.0e-12)
                self.assertLess(np.max(np.abs(full.apply(spinors) - spinors * levels[:6])), 1.0e-10)
                if plan.partner[sector] is not None:
                    values.append(levels)
                    partners = plan.partner_spinors(spinors)
                    self.assertLess(np.max(np.abs(full.apply(partners) - partners * levels[:6])), 1.0e-10)
                    # the partners are orthogonal to the states of the sector solved
                    self.assertLess(np.max(np.abs(spinors.conj().T @ partners)), 1.0e-12)
                    # the density of the partners is the image under h of that of the columns solved
                    density = np.sum(np.abs(spinors) ** 2, axis=1)
                    n = system.grid.size
                    image = plan.image_density(density[:n] + density[n:])
                    partner_density = np.sum(np.abs(partners) ** 2, axis=1)
                    np.testing.assert_allclose(partner_density[:n] + partner_density[n:], image, atol=1.0e-14)
            union = np.sort(np.concatenate(values))
            self.assertEqual(union.size, whole.size)
            self.assertLess(np.max(np.abs(union - whole)), 1.0e-10)
        self.assertGreater(split, 0)
        return plans

    def test_atom_of_the_cubic_group(self) -> None:
        plans = self.check("atom", {2})
        gamma = plans[0]
        # Gamma: {E, i, C2z, sigma_z}, four sectors in two Kramers pairs of time reversal
        self.assertEqual(gamma.order, 4)
        self.assertEqual(gamma.antiunitary, 0)
        self.assertEqual(len(gamma.solved), 2)
        # (0, 0, 1/3): C2z, paired by the inversion
        along = plans[5]
        self.assertEqual(along.order, 2)
        self.assertEqual(len(along.solved), 1)

    def test_pair_along_z(self) -> None:
        self.check("pair", {2})

    def test_magnetization_along_z(self) -> None:
        for case in ("atom", "pair"):
            plans = self.check(case, {2}, magnetic=True)
            self.assertEqual(plans[0].order, 4)
            self.assertEqual(len(plans[0].solved), 4)

    def test_partners_turned_by_c2_or_mirror(self) -> None:
        from parsec_python.acceleration.SCF.kpoint_spinor_sectors import _IDENTITY_TYPE, _proper_type

        for case, axis in (("c2hx", 0), ("atom", 2)):
            with _turned_partners():
                plans = self.check(case, {axis})
            turned = [
                plan for plan in plans[1:] if plan is not None and plan.antiunitary is not None
                and _proper_type(plan.cell.matrices[plan.antiunitary]) not in (None, _IDENTITY_TYPE)
            ]
            # every k-point but Gamma of the star pairs its sectors through C2a or sigma_a
            self.assertEqual(len(turned), 6, case)
            for plan in turned:
                self.assertGreater(np.max(np.abs(plan._spin_turn - np.eye(2))), 0.5)

    def test_mirror_normal_to_y(self) -> None:
        # (the crystal group exchanges the two atoms by an operation that does not map this grid: no symmetrized
        # density; the mirror holds at self-consistency, the set of k-points mapping onto itself)
        plans = self.check("mirror", {1}, symmetrized=False)
        # no operation reverses (0, 0, 1/3) with the mirror: both sectors are solved
        self.assertEqual(len(plans[5].solved), 2)


class SpinorSectorSCFTests(unittest.TestCase):
    @staticmethod
    def _run(case: str, value: str, symmetrized: bool):
        k_points, weighting = _k_points()
        weights = np.full(len(k_points), 1.0 / len(k_points))
        with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: value}):
            return run_self_consistent_soc_kpoints(
                _system(case, symmetrized), k_points, weights, compute_forces=True
            )

    def check(self, case: str, symmetrized: bool = True) -> None:
        whole = self._run(case, "off", symmetrized)
        sectors = self._run(case, "on", symmetrized)
        self.assertTrue(whole.converged and sectors.converged)
        self.assertNotIn("eig_spinor_sector_tasks", whole.timings)
        self.assertGreater(sectors.timings["eig_spinor_sector_kpoints"], 0)
        self.assertLess(abs(sectors.energies.total - whole.energies.total), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.density - whole.density)), 1.0e-9)
        occupied = whole.occupations > 1.0e-8
        np.testing.assert_allclose(
            np.sort(sectors.eigenvalues[sectors.occupations > 1.0e-8]), np.sort(whole.eigenvalues[occupied]),
            atol=1.0e-9,
        )
        self.assertLess(np.max(np.abs(sectors.nonlocal_forces - whole.nonlocal_forces)), 1.0e-8)
        self.assertAlmostEqual(float(np.sum(sectors.state_moments)), float(np.sum(whole.state_moments)), places=8)
        spinors = sectors.spinors
        self.assertEqual(spinors.shape, whole.spinors.shape)
        np.testing.assert_allclose(np.linalg.norm(spinors, axis=0), 1.0, atol=1.0e-12)

    def test_pair_along_z(self) -> None:
        self.check("pair")

    def test_magnetic_atom(self) -> None:
        k_points, _ = _k_points()
        weights = np.full(len(k_points), 1.0 / len(k_points))
        system = _system("atom", True, 1.0e-11, True)
        results = {}
        for value in ("off", "on"):
            with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: "on", SPIN_SWITCH: value}):
                results[value] = run_self_consistent_soc_kpoints_spin_polarized(
                    system, k_points, weights, compute_forces=True
                )
        whole, sectors = results["off"], results["on"]
        self.assertTrue(whole.converged and sectors.converged)
        self.assertNotIn("eig_spinor_sector_tasks", whole.timings)
        self.assertGreater(sectors.timings["eig_spinor_sector_kpoints"], 0)
        self.assertGreater(abs(whole.magnetic_moment), 1.0e-3)
        self.assertLess(abs(sectors.energies.total - whole.energies.total), 1.0e-9)
        self.assertLess(abs(sectors.magnetic_moment - whole.magnetic_moment), 1.0e-8)
        self.assertLess(np.max(np.abs(sectors.density - whole.density)), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.nonlocal_forces - whole.nonlocal_forces)), 1.0e-8)
        self.assertAlmostEqual(float(np.sum(sectors.state_moments)), float(np.sum(whole.state_moments)), places=7)

    def test_partners_turned_by_c2_or_mirror(self) -> None:
        k_points, _ = _k_points()
        weights = np.full(len(k_points), 1.0 / len(k_points))
        system = _system("c2hx")
        whole = self._run("c2hx", "off", True)
        with _turned_partners():
            with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: "on"}):
                sectors = run_self_consistent_soc_kpoints(system, k_points, weights, compute_forces=True)
        self.assertTrue(whole.converged and sectors.converged)
        self.assertLess(abs(sectors.energies.total - whole.energies.total), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.density - whole.density)), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.nonlocal_forces - whole.nonlocal_forces)), 1.0e-8)

    def test_states_carried_to_the_next_solver(self) -> None:
        from parsec_python.SCF.kpoints_soc import _spinor_solver

        system = _system("pair")
        k_points, weighting = _k_points()
        states = int(system.input.scf.number_of_states)

        def accumulate(task, vectors, occupations, xp=np):
            return {"density": np.zeros(1)}

        potentials = {"V": np.asarray(system.ionic_potential, dtype=np.float64)}
        degree = int(system.input.eigensolver.filter_degree)
        with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: "on"}):
            geometry = _sector_geometry(system, k_points, weighting, False)
            first, _ = _spinor_solver(system, k_points, states, accumulate, geometry=geometry)
            with first:
                self.assertIsInstance(first, SpinorSectorKPointSolver)
                levels = first.solve(potentials, degree)
                levels = first.solve(potentials, degree)
                carried = first.export_states()
                started = first.stats()
            second, _ = _spinor_solver(system, k_points, states, accumulate, geometry=geometry)
            with second:
                second.import_states(carried)
                again = second.solve(potentials, degree)
                stats = second.stats()
            # a solver of other sectors (here: one task per k-point) does not take them
            with mock.patch.dict(os.environ, {SWITCH: "off"}):
                plain, _ = _spinor_solver(system, k_points, states, accumulate, geometry=None)
            with plain:
                plain.import_states(carried)
        self.assertGreater(started.get("first_solves", 0.0), 0.0)
        self.assertEqual(stats.get("first_solves", 0.0), 0.0)
        for previous, current in zip(levels, again):
            np.testing.assert_allclose(current, previous, atol=1.0e-8)

    def test_mirror_normal_to_y_without_symmetrized_density(self) -> None:
        # the operations hold at self-consistency only (the set of k-points maps onto itself): the potential averaged
        # over the orbits, Kramers pairs only at Gamma
        self.check("mirror", symmetrized=False)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value in ("off", "auto", "on"):
            with mock.patch.dict(os.environ, {SWITCH: value}):
                self.assertEqual(soc_kpoint_sectors_setting(), value)
        with mock.patch.dict(os.environ, {SWITCH: "maybe"}):
            with self.assertRaises(ValueError):
                soc_kpoint_sectors_setting()
            from parsec_python.acceleration.SCF.kpoint_sectors import check_kpoint_sector_switches

            with self.assertRaises(ValueError):
                check_kpoint_sector_switches()

    def test_off_keeps_the_job_of_before(self) -> None:
        system = _system("atom")
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            self.assertIsNone(_sector_geometry(system, k_points, weighting, False))

    def test_auto_keeps_small_bases_whole(self) -> None:
        system = _system("atom")
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {SWITCH: "auto"}):
            plans, notes = _plans(system, k_points, weighting, int(system.input.scf.number_of_states))
        self.assertIsNone(plans)
        self.assertIn("below the floor", notes[-1])

    def test_the_solver_of_the_sectors_is_made(self) -> None:
        from parsec_python.SCF.kpoint_solver import make_kpoint_solver

        system = _system("atom")
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: "on"}):
            geometry = _sector_geometry(system, k_points, weighting, False)
            job = KPointJob(len(k_points), 20, lambda *a: None, lambda *a: None, lambda *a, **k: {}, geometry=geometry)
            solver = make_kpoint_solver(job, problem_size=system.grid.size)
            try:
                self.assertIsInstance(solver, SpinorSectorKPointSolver)
                self.assertIn("double-group spinor sectors", solver.kpoint_symmetry_detection)
            finally:
                solver.close()
        # auto: k-points dealt over two devices of the process keep one task per k-point
        with mock.patch.dict(os.environ, {**_DEVICE, SWITCH: "auto", "PARSEC_KPOINT_WORKERS": "2"}):
            solver = make_kpoint_solver(job, problem_size=system.grid.size)
            try:
                self.assertNotIsInstance(solver, SpinorSectorKPointSolver)
                self.assertIn("devices", solver.kpoint_symmetry_detection)
            finally:
                solver.close()
        # the CPU route keeps one task per k-point
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "cpu", SWITCH: "on"}):
            solver = make_kpoint_solver(job, problem_size=system.grid.size)
            try:
                self.assertNotIsInstance(solver, SpinorSectorKPointSolver)
                self.assertIn("device solver", solver.kpoint_symmetry_detection)
            finally:
                solver.close()


if __name__ == "__main__":
    unittest.main()
