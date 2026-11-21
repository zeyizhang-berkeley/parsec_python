"""Double-group sectors of the self-consistent spin-orbit loop (PARSEC_SOC_SECTORS), on the host.

The sector operator of :mod:`parsec_python.acceleration.SCF.soc_sectors` runs here with NumPy as the array backend
(the code the GPU runs).  On small grids its spectrum is checked against the dense whole-grid spinor operator (every
eigenvalue once per Kramers pair), its unfolded states against the eigenvectors of that operator, and the SCF of
the sector against the whole-grid loop.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    MixingSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Eigensolvers.subspace import SubspaceSettings
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.Eigensolvers.chebff_spinor import solve_spinor_eigval
from parsec_python.acceleration.Hamiltonian.device_bloch import NumpyArrayBackend
from parsec_python.acceleration.SCF.self_consistent_soc import (
    DeviceSpinorHamiltonianBuilder,
    run_self_consistent_soc,
)
from parsec_python.acceleration.SCF.soc_sectors import (
    SWITCH,
    SectorRoute,
    SectorSpinors,
    detect_sector_plan,
    resolve_soc_sectors,
    soc_sectors_setting,
    whole_grid_spinor_entries,
)

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
_AU = _EXAMPLE / "Au_POTRE.DAT"
_H = _EXAMPLE / "H_POTRE.DAT"


def _problem(atoms, *, spacing=0.7, radius=4.5, states=12, iterations=5) -> SinglePointInput:
    potentials = {"Au": SpeciesPotential(_AU, 0, spin_orbit=True)}
    if any(atom.symbol == "H" for atom in atoms):
        potentials["H"] = SpeciesPotential(_H, 0, spin_orbit=False)
    return SinglePointInput(
        atoms=atoms,
        pseudopotentials=potentials,
        grid=GridSettings(spacing=spacing, radius=radius, expansion_order=6),
        scf=SCFSettings(max_iterations=iterations, number_of_states=states, convergence_criterion=1.0e-4),
        hartree=HartreeSettings(multipole_order=4),
        mixing=MixingSettings(method="anderson", parameter=0.25),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _prepared(atoms, **keywords):
    problem = _problem(atoms, **keywords)
    system = prepare_single_point(problem)
    projectors = tuple(
        build_spin_orbit_projectors(system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials)
    )
    return system, projectors


class SectorOperatorTests(unittest.TestCase):
    """The reduced operator against the dense whole-grid spinor operator."""

    CASES = {
        # D2h: the sector of C2z made real by T C2x
        "atom": ([Atom("Au", [0.0, 0.0, 0.0])], True, "C2z"),
        # C2v along x: the sector of sigma_z made real by T C2x
        "AuH along x": ([Atom("Au", [1.4, 0.0, 0.0]), Atom("H", [-1.4, 0.0, 0.0])], True, "sigma_z"),
        # C2v along z: the sector of C2z made real by T sigma_x
        "AuH along z": ([Atom("Au", [0.0, 0.0, 1.4]), Atom("H", [0.0, 0.0, -1.4])], True, "C2z"),
        # C2v along y: the sector of sigma_z made real by T C2y (the phases of an h about y)
        "AuH along y": ([Atom("Au", [0.0, 1.4, 0.0]), Atom("H", [0.0, -1.4, 0.0])], True, "sigma_z"),
        # Cs (sigma_z only): a complex sector
        "AuH in the xy plane": ([Atom("Au", [0.9, 0.7, 0.0]), Atom("H", [-0.9, -0.7, 0.0])], False, "sigma_z"),
        # Cs (sigma_x only): a complex sector with the spin along x
        "AuH in the yz plane": ([Atom("Au", [0.0, 0.5, 0.7]), Atom("H", [0.0, -0.4, -0.3])], False, "sigma_x"),
        # C2h along y: a complex sector with the spin along y
        "Au2 in the xz plane": ([Atom("Au", [0.6, 0.0, 0.4]), Atom("Au", [-0.6, 0.0, -0.4])], False, "C2y"),
    }

    def check(self, atoms, real: bool, generator: str) -> None:
        system, projectors = _prepared(atoms)
        backend = NumpyArrayBackend()
        scalar = system.hamiltonian(system.ionic_potential)
        full = DeviceSpinorHamiltonianBuilder(backend)(scalar, projectors)
        dense = full.apply(np.eye(full.shape[0], dtype=complex))
        full_values = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))

        plan, reason = detect_sector_plan(system.grid, system.atoms, parity=False)
        self.assertIsNotNone(plan, reason)
        self.assertEqual(plan.sector_count, 1)
        self.assertEqual(plan.real, real)
        self.assertEqual(plan.generators[0], generator)
        self.assertEqual(plan.dimension, system.grid.size)
        route = SectorRoute(plan, backend, projectors)
        operator = route.hamiltonian(scalar, projectors)
        self.assertEqual(bool(getattr(operator, "real_arithmetic", False)), real)
        size = operator.shape[0]
        reduced = operator.apply(np.eye(size) if real else np.eye(size, dtype=complex))
        self.assertEqual(reduced.dtype, np.float64 if real else np.complex128)
        scale = float(np.abs(reduced).max())
        self.assertLess(float(np.abs(reduced - reduced.conj().T).max()), 1.0e-13 * scale)
        values, vectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
        # every eigenvalue of the whole grid once per Kramers pair
        np.testing.assert_allclose(np.repeat(values, 2), full_values, atol=1.0e-11 * scale)
        # the unfolded states and their partners are orthonormal eigenvectors of the whole-grid operator
        spinors = SectorSpinors.single(operator.parts, vectors[:, :5], 9).full_spinors(chunk=2)
        self.assertEqual(spinors.shape, (2 * system.grid.size, 9))
        np.testing.assert_allclose(spinors.conj().T @ spinors, np.eye(9), atol=1.0e-12)
        residual = full.apply(spinors) - spinors * np.repeat(values[:5], 2)[None, :9]
        self.assertLess(float(np.abs(residual).max()), 1.0e-11 * scale)
        # the density of the pairs is that of the unfolded states
        occupations = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.1])
        sector_density = SectorSpinors.single(operator.parts, vectors[:, :5], 9).density(occupations, 0.3, chunk=2)
        expected = np.sum(
            (np.abs(spinors[: system.grid.size]) ** 2 + np.abs(spinors[system.grid.size :]) ** 2) * occupations,
            axis=1,
        ) / 0.3
        np.testing.assert_allclose(sector_density, expected, rtol=1.0e-12, atol=1.0e-14)
        # with a centre of inversion: the even and odd sectors hold those eigenvalues between them
        split, _ = detect_sector_plan(system.grid, system.atoms, parity=True)
        if "i" not in split.group:
            self.assertEqual(split.sector_count, 1)
            return
        self.assertEqual(split.sector_count, 2)
        operators = SectorRoute(split, backend, projectors).hamiltonian(scalar, projectors).sectors
        both = []
        for sector in operators:
            rows = sector.shape[0]
            matrix = sector.apply(np.eye(rows) if real else np.eye(rows, dtype=complex))
            self.assertLess(float(np.abs(matrix - matrix.conj().T).max()), 1.0e-13 * scale)
            both.append(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)))
        self.assertEqual(sum(part.size for part in both), size)
        np.testing.assert_allclose(np.sort(np.concatenate(both)), values, atol=1.0e-11 * scale)

    def test_cases(self) -> None:
        for name, (atoms, real, generator) in self.CASES.items():
            with self.subTest(name):
                self.check(atoms, real, generator)

    def test_a_cluster_without_a_two_fold_operation_keeps_the_whole_grid(self) -> None:
        system, _ = _prepared([Atom("Au", [0.6, 0.3, 0.0]), Atom("H", [-0.6, 0.9, 0.5])])
        plan, reason = detect_sector_plan(system.grid, system.atoms)
        self.assertIsNone(plan)
        self.assertIn("no two-fold rotation or mirror", reason)


class RealSolverTests(unittest.TestCase):
    """``solve_spinor_eigval`` on a real symmetric operator (``real_arithmetic``)."""

    class _Real:
        real_arithmetic = True

        def __init__(self, matrix):
            self.matrix = matrix
            self.shape = matrix.shape

        def apply(self, block):
            assert block.dtype == np.float64
            return self.matrix @ block

    def test_real_blocks_converge_to_the_lowest_states(self) -> None:
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(150, 150))
        matrix = 0.5 * (raw + raw.T)
        exact = np.linalg.eigvalsh(matrix)
        settings = EigvalSettings(
            safety_buffer=4,
            initial_method="chebff",
            chebff=ChebFFSettings(polynomial_degree=24, filter_cycles=4, lanczos_steps=10, block_size=4),
            subspace=SubspaceSettings(polynomial_degree=14, degree_delta=3, block_size=4),
        )
        operator = self._Real(matrix)
        result = solve_spinor_eigval(operator, 6, settings=settings, xp=np)
        self.assertEqual(result.state.vectors.dtype, np.float64)
        state = result.state
        for _ in range(6):
            result = solve_spinor_eigval(operator, 6, settings=settings, state=state, xp=np)
            state = result.state
        self.assertEqual(state.vectors.dtype, np.float64)
        np.testing.assert_allclose(result.eigenvalues, exact[:6], atol=1.0e-5)


class SectorSCFTests(unittest.TestCase):
    """The SCF of the sectors against the whole-grid loop (Au atom: D2h, NumPy backend)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system, cls.projectors = _prepared(
            [Atom("Au", [0.0, 0.0, 0.0])], spacing=0.6, radius=6.0, states=20, iterations=40
        )
        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            cls.whole = run_self_consistent_soc(cls.system, cls.projectors, backend=NumpyArrayBackend())
        cls.notes = {}
        cls.sectors = {}
        for setting in ("kramers", "on"):
            cls.notes[setting] = []
            with mock.patch.dict(os.environ, {SWITCH: setting}):
                cls.sectors[setting] = run_self_consistent_soc(
                    cls.system, cls.projectors, backend=NumpyArrayBackend(), note=cls.notes[setting].append
                )

    def test_same_energy_states_and_density(self) -> None:
        self.assertTrue(self.whole.converged)
        for setting, sector in self.sectors.items():
            with self.subTest(setting):
                self.assertTrue(sector.converged)
                self.assertAlmostEqual(sector.energies.total, self.whole.energies.total, delta=1.0e-7)
                np.testing.assert_allclose(sector.eigenvalues, self.whole.eigenvalues, atol=1.0e-4)
                np.testing.assert_allclose(sector.density, self.whole.density, atol=1.0e-4)
                np.testing.assert_allclose(sector.occupations, self.whole.occupations, atol=1.0e-6)

    def test_spinors_are_the_whole_grid_states(self) -> None:
        for setting, sector in self.sectors.items():
            with self.subTest(setting):
                spinors = sector.spinors
                self.assertIsInstance(spinors, np.ndarray)
                self.assertEqual(spinors.shape, self.whole.spinors.shape)
                np.testing.assert_allclose(spinors.conj().T @ spinors, np.eye(spinors.shape[1]), atol=1.0e-10)

    def test_half_the_columns_are_filtered(self) -> None:
        sector = self.sectors["kramers"]
        self.assertLess(sector.timings["eig_application_columns"], 0.6 * self.whole.timings["eig_application_columns"])
        self.assertIn("soc_sector_setup", sector.timings)
        self.assertNotIn("soc_sector_setup", self.whole.timings)
        self.assertEqual(len(self.notes["kramers"]), 1)
        self.assertIn("sector +i of C2z", self.notes["kramers"][0])
        self.assertIn("split by the parity of i", self.notes["on"][0])
        self.assertIn("eig_sector_pairs", self.sectors["on"].timings)


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class DeviceSectorTests(unittest.TestCase):
    """The sector operator, density and unfolded states on the GPU against the same code with NumPy."""

    def test_cupy_matches_numpy(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        cases = {
            "real (AuH along x)": [Atom("Au", [1.4, 0.0, 0.0]), Atom("H", [-1.4, 0.0, 0.0])],
            "complex, spin along y, parity (Au2 in the xz plane)": [
                Atom("Au", [0.6, 0.0, 0.4]),
                Atom("Au", [-0.6, 0.0, -0.4]),
            ],
        }
        rng = np.random.default_rng(3)
        for name, atoms in cases.items():
            with self.subTest(name):
                system, projectors = _prepared(atoms)
                scalar = system.hamiltonian(system.ionic_potential)
                plan, _ = detect_sector_plan(system.grid, system.atoms, parity=True)
                host = SectorRoute(plan, NumpyArrayBackend(), projectors).hamiltonian(scalar, projectors)
                device = SectorRoute(plan, CupyArrayBackend(0), projectors).hamiltonian(scalar, projectors)
                host_sectors = getattr(host, "sectors", [host])
                device_sectors = getattr(device, "sectors", [device])
                for host_operator, device_operator in zip(host_sectors, device_sectors):
                    rows = host_operator.shape[0]
                    block = rng.normal(size=(rows, 7))
                    if not plan.real:
                        block = block + 1j * rng.normal(size=(rows, 7))
                    expected = host_operator.apply(block)
                    found = cp.asnumpy(device_operator.apply(cp.asarray(block)))
                    scale = float(np.abs(expected).max())
                    np.testing.assert_allclose(found, expected, atol=1.0e-12 * scale)
                    occupations = np.linspace(1.0, 0.1, 13)
                    host_states = SectorSpinors.single(host_operator.parts, block, 13)
                    device_states = SectorSpinors.single(device_operator.parts, cp.asarray(block), 13)
                    np.testing.assert_allclose(
                        device_states.density(occupations, 0.3, chunk=3),
                        host_states.density(occupations, 0.3, chunk=3),
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    )
                    np.testing.assert_allclose(
                        device_states.full_spinors(chunk=3), host_states.full_spinors(chunk=3), atol=1.0e-13
                    )
                if len(host_sectors) == 2:
                    # the pairs of both parity sectors interleaved by place (gaps, unused columns, a missing pair)
                    blocks = []
                    for sector in host_sectors:
                        values = rng.normal(size=(sector.shape[0], 5))
                        blocks.append(values if plan.real else values + 1j * rng.normal(size=values.shape))
                    places = [np.array([0, 2, 3, -1, 6]), np.array([1, -1, 4, 5, -1])]
                    for chunk in (2, 5, 64):
                        host_states = SectorSpinors([s.parts for s in host_sectors], blocks, places, 15)
                        device_states = SectorSpinors(
                            [s.parts for s in device_sectors], [cp.asarray(b) for b in blocks], places, 15
                        )
                        np.testing.assert_allclose(
                            device_states.full_spinors(chunk), host_states.full_spinors(chunk), atol=1.0e-13
                        )


def _column_by_column(spinors: SectorSpinors, chunk: int) -> np.ndarray:
    """``SectorSpinors.full_spinors`` as it was first written (one column at a time): the reference of the unfold."""
    n = spinors.parts[0].grid_size
    output = np.zeros((2 * n, spinors.count), dtype=np.complex128)
    for parts, vectors, places in zip(spinors.parts, spinors.vectors, spinors.places):
        for c0 in range(0, int(vectors.shape[1]), chunk):
            c1 = min(c0 + chunk, int(vectors.shape[1]))
            unfolded = parts.unfold_block(np.ascontiguousarray(vectors[:, c0:c1]))
            partners = np.concatenate((-np.conj(unfolded[n:]), np.conj(unfolded[:n])))
            for local in range(c1 - c0):
                place = int(places[c0 + local])
                if place < 0:
                    continue
                if 2 * place < spinors.count:
                    output[:, 2 * place] = unfolded[:, local]
                if 2 * place + 1 < spinors.count:
                    output[:, 2 * place + 1] = partners[:, local]
    return output


class UnfoldTests(unittest.TestCase):
    """The whole-grid spinors of a sector solve (interleaved on the device, one copy per chunk) against the
    column-by-column unfold: one sector and the parity sectors, odd and even counts, places with gaps and unused
    columns, chunks of one column to more than all."""

    def test_same_spinors_as_column_by_column(self) -> None:
        rng = np.random.default_rng(11)
        cases = {
            "real (AuH along x)": [Atom("Au", [1.4, 0.0, 0.0]), Atom("H", [-1.4, 0.0, 0.0])],
            "complex, parity (Au2 in the xz plane)": [Atom("Au", [0.6, 0.0, 0.4]), Atom("Au", [-0.6, 0.0, -0.4])],
        }
        backend = NumpyArrayBackend()
        for name, atoms in cases.items():
            system, projectors = _prepared(atoms)
            scalar = system.hamiltonian(system.ionic_potential)
            for parity in (False, True):
                plan, _ = detect_sector_plan(system.grid, system.atoms, parity=parity)
                operator = SectorRoute(plan, backend, projectors).hamiltonian(scalar, projectors)
                sectors = getattr(operator, "sectors", [operator])

                def block(rows, columns):
                    values = rng.normal(size=(rows, columns))
                    return values if plan.real else values + 1j * rng.normal(size=(rows, columns))

                if len(sectors) == 1:
                    layouts = {
                        count: SectorSpinors.single(sectors[0].parts, block(sectors[0].shape[0], 8), count)
                        for count in (13, 14, 16)  # the last pair cut, every pair, more columns than pairs
                    }
                else:
                    # 7 pairs of 13 states between two sectors, in order with gaps, unused columns between them;
                    # with 15 states the eighth pair has no column (its state stays zero)
                    vectors = [block(sectors[0].shape[0], 5), block(sectors[1].shape[0], 5)]
                    layouts = {
                        count: SectorSpinors(
                            parts=[sector.parts for sector in sectors],
                            vectors=vectors,
                            places=[np.array([0, 2, 3, -1, 6]), np.array([1, -1, 4, 5, -1])],
                            count=count,
                        )
                        for count in (13, 15)
                    }
                for count, spinors in layouts.items():
                    for chunk in (1, 2, 3, 64):
                        with self.subTest(name, parity=parity, count=count, chunk=chunk):
                            found = spinors.full_spinors(chunk)
                            self.assertEqual(found.shape, (2 * system.grid.size, count))
                            np.testing.assert_array_equal(found, _column_by_column(spinors, 4))


class OrbitalSymmetryTests(unittest.TestCase):
    """What the report says of the orbitals of a spin-orbit run (the ``orbital_symmetry`` line)."""

    def test_the_line_names_the_sectors_the_loop_solves(self) -> None:
        from parsec_python.acceleration.SCF.soc_sectors import soc_orbital_symmetry

        system, _ = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=1)
        for setting, parity in (("kramers", False), ("on", True)):
            with self.subTest(setting):
                text, reason = soc_orbital_symmetry(system.grid, system.atoms, setting)
                plan, _ = detect_sector_plan(system.grid, system.atoms, parity=parity)
                self.assertEqual(reason, "")
                self.assertTrue(text.startswith("spin-orbit double-group sector +i of C2z made real by T C2x"))
                self.assertIn(f"PARSEC_SOC_SECTORS={setting}", text)
                self.assertIn(", ".join(plan.group), text)
                self.assertEqual("split by the parity of i" in text, plan.sector_count > 1)
        generic, _ = _prepared([Atom("Au", [0.6, 0.3, 0.0]), Atom("H", [-0.6, 0.9, 0.5])], iterations=1)
        text, reason = soc_orbital_symmetry(generic.grid, generic.atoms, "kramers")
        self.assertIn("no two-fold rotation or mirror", reason)
        self.assertTrue(text.startswith("full grid (PARSEC_SOC_SECTORS=kramers: "))

    def test_backend_record_of_the_cli(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.cli import _soc_orbital_symmetry
        from parsec_python.acceleration.models import BackendInfo

        info = BackendInfo(
            requested="auto",
            selected="cupy",
            details=(("gpu_stencil_kernel", "pipelined"), ("orbital_symmetry", "full grid (symmetry disabled)")),
        )
        atom, _ = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=1)
        generic, _ = _prepared([Atom("Au", [0.6, 0.3, 0.0]), Atom("H", [-0.6, 0.9, 0.5])], iterations=1)

        def record(system, setting, symmetry_on=False):
            found = _soc_orbital_symmetry(
                SimpleNamespace(backend_info=info, grid=system.grid, atoms=system.atoms), setting, symmetry_on
            )
            self.assertEqual([key for key, _ in found.details], ["gpu_stencil_kernel", "orbital_symmetry"])
            return dict(found.details)["orbital_symmetry"]

        self.assertEqual(record(atom, "off"), "full grid (self-consistent spin-orbit coupling: PARSEC_SOC_SECTORS=off)")
        self.assertIn("sector +i of C2z", record(atom, "kramers", symmetry_on=True))
        self.assertIn("no two-fold rotation or mirror", record(generic, "on"))
        # --symmetry on promises a reduction: refused before the SCF where the sectors cannot be used
        with self.assertRaises(ValueError) as caught:
            record(generic, "kramers", symmetry_on=True)
        self.assertIn("no two-fold rotation or mirror", str(caught.exception))


_CLI_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3

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
   {au}
End atom_coord

Atom_Type: H
SO_PSP: .false.
Local_Component: s
Begin atom_coord
   {h}
End atom_coord

Correlation_Type: CA
"""


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class CommandLineTests(unittest.TestCase):
    """The accelerated command line with PARSEC_SOC_SECTORS: --symmetry on, and what parsec.out says of the
    symmetry (the block of the point group, the orbitals)."""

    ALONG_X = {"au": "2.89 0 0", "h": "0 0 0"}  # C2v
    GENERIC = {"au": "1.2 0.7 0.4", "h": "0 0 0"}  # the identity alone

    def run_cli(self, setting: str, arguments=(), atoms=None, extra: str = ""):
        import contextlib
        import io
        import shutil
        import tempfile

        from parsec_python.acceleration.cli import main as accelerated_main

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copy(_AU, directory / "Au_POTRE.DAT")
        shutil.copy(_H, directory / "H_POTRE.DAT")
        path = directory / "parsec.in"
        path.write_text(_CLI_INPUT.format(extra=extra, **(atoms or self.ALONG_X)), encoding="utf-8")
        errors = io.StringIO()
        with mock.patch.dict(os.environ, {SWITCH: setting}), contextlib.redirect_stderr(errors):
            code = accelerated_main([str(path), "--quiet", "--no-archive", *arguments])
        out = directory / "parsec.out"
        return code, (out.read_text(encoding="utf-8") if out.exists() else ""), errors.getvalue()

    def test_symmetry_on_runs_the_sectors(self) -> None:
        code, log, errors = self.run_cli("kramers", ["--symmetry", "on"])
        self.assertEqual(code, 0, errors)
        self.assertIn("orbital_symmetry = spin-orbit double-group sector +i of sigma_z made real by T C2x", log)
        self.assertIn("Spin-orbit sectors (PARSEC_SOC_SECTORS=kramers)", log)
        self.assertNotIn("Ignore_Symmetry flag", log)
        self.assertIn("ignoresym= F", log)

    def test_the_command_line_overrides_ignore_symmetry(self) -> None:
        # as for every other run, an explicit mode on the command line outranks Ignore_Symmetry
        ignore = "Ignore_Symmetry: .true.\n"
        code, log, errors = self.run_cli("kramers", ["--symmetry", "on"], extra=ignore)
        self.assertEqual(code, 0, errors)
        self.assertIn("orbital_symmetry = spin-orbit double-group sector +i of sigma_z made real by T C2x", log)
        self.assertIn("Spin-orbit sectors (PARSEC_SOC_SECTORS=kramers)", log)
        self.assertNotIn("Ignore_Symmetry flag", log)
        self.assertNotIn("the spinors stay on the whole grid", log)
        code, _, errors = self.run_cli("off", ["--symmetry", "on"], extra=ignore)
        self.assertEqual(code, 2)
        self.assertIn("PARSEC_SOC_SECTORS", errors)

    def test_symmetry_on_without_the_sectors_is_refused(self) -> None:
        code, _, errors = self.run_cli("off", ["--symmetry", "on"])
        self.assertEqual(code, 2)
        self.assertIn("PARSEC_SOC_SECTORS", errors)
        code, log, _ = self.run_cli("kramers", ["--symmetry", "on"], atoms=self.GENERIC)
        self.assertEqual(code, 1)
        self.assertIn("Calculation failed: ValueError: symmetry='on' requested", log)
        self.assertIn("no two-fold rotation or mirror", log)
        self.assertNotIn("Self-consistent SOC iter", log)

    def test_report_of_the_whole_grid(self) -> None:
        code, log, errors = self.run_cli("off")
        self.assertEqual(code, 0, errors)
        self.assertNotIn("Ignore_Symmetry flag", log)
        self.assertIn("orbital_symmetry = full grid (self-consistent spin-orbit coupling: PARSEC_SOC_SECTORS=off)", log)
        code, log, errors = self.run_cli("kramers", extra="Ignore_Symmetry: .true.\n")
        self.assertEqual(code, 0, errors)
        self.assertIn("Symmetry suppressed by the Ignore_Symmetry flag.", log)
        self.assertIn("orbital_symmetry = full grid (symmetry disabled)", log)
        self.assertIn("the spinors stay on the whole grid", log)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("off", "off"), ("ON", "on"), ("kramers", "kramers")):
            with mock.patch.dict(os.environ, {SWITCH: value}):
                self.assertEqual(soc_sectors_setting(), expected)
        # the default since 2026-11-21
        with mock.patch.dict(os.environ):
            os.environ.pop(SWITCH, None)
            self.assertEqual(soc_sectors_setting(), "auto")
        with mock.patch.dict(os.environ, {SWITCH: "yes"}):
            with self.assertRaises(ValueError):
                soc_sectors_setting()

    def test_an_input_without_symmetry_keeps_the_whole_grid(self) -> None:
        system, projectors = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=1)
        notes = []
        with mock.patch.dict(os.environ, {SWITCH: "on"}), mock.patch(
            "parsec_python.acceleration.SCF.self_consistent_soc._sector_route",
            side_effect=AssertionError("no sectors without symmetry"),
        ):
            result = run_self_consistent_soc(
                system, projectors, backend=NumpyArrayBackend(), note=notes.append, symmetry=False
            )
        self.assertNotIn("soc_sector_setup", result.timings)
        self.assertEqual(len(notes), 1)
        self.assertIn("the spinors stay on the whole grid", notes[0])

    def test_auto_takes_the_kramers_sector_from_the_floor_on(self) -> None:
        from parsec_python.acceleration.SCF import soc_sectors

        system, _ = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=1)
        states = 12 + system.input.eigensolver.subspace_buffer
        self.assertEqual(whole_grid_spinor_entries(system), 2 * system.grid.size * states)
        self.assertLess(whole_grid_spinor_entries(system), soc_sectors.MINIMUM_ENTRIES)
        route, reason = resolve_soc_sectors("auto", system)
        self.assertEqual(route, "off")
        self.assertIn("below the floor of auto", reason)
        # --symmetry on asks for a reduction at every size; an explicit value is its own route
        self.assertEqual(resolve_soc_sectors("auto", system, forced=True), ("kramers", ""))
        for setting in ("off", "kramers", "on"):
            self.assertEqual(resolve_soc_sectors(setting, system), (setting, ""))
        with mock.patch.object(soc_sectors, "MINIMUM_ENTRIES", whole_grid_spinor_entries(system)):
            self.assertEqual(resolve_soc_sectors("auto", system), ("kramers", ""))

    def test_auto_below_the_floor_is_the_whole_grid(self) -> None:
        from parsec_python.acceleration.SCF import soc_sectors

        system, projectors = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=3)
        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            whole = run_self_consistent_soc(system, projectors, backend=NumpyArrayBackend())
        notes = []
        with mock.patch.dict(os.environ, {SWITCH: "auto"}), mock.patch(
            "parsec_python.acceleration.SCF.self_consistent_soc._sector_route",
            side_effect=AssertionError("no sectors below the floor"),
        ):
            below = run_self_consistent_soc(system, projectors, backend=NumpyArrayBackend(), note=notes.append)
        self.assertEqual(below.energies.total, whole.energies.total)
        np.testing.assert_array_equal(below.eigenvalues, whole.eigenvalues)
        self.assertEqual(len(notes), 1)
        self.assertIn("PARSEC_SOC_SECTORS=auto", notes[0])
        self.assertIn("the spinors stay on the whole grid", notes[0])
        # no symmetry: the whole grid without a note, as the former default
        notes = []
        with mock.patch.dict(os.environ, {SWITCH: "auto"}):
            run_self_consistent_soc(system, projectors, backend=NumpyArrayBackend(), note=notes.append, symmetry=False)
        self.assertEqual(notes, [])
        # from the floor on: the Kramers sector, named as the route of auto
        notes = []
        with mock.patch.dict(os.environ, {SWITCH: "auto"}), mock.patch.object(soc_sectors, "MINIMUM_ENTRIES", 1):
            sector = run_self_consistent_soc(system, projectors, backend=NumpyArrayBackend(), note=notes.append)
        self.assertIn("soc_sector_setup", sector.timings)
        self.assertEqual(len(notes), 1)
        self.assertIn("Spin-orbit sectors (PARSEC_SOC_SECTORS=auto)", notes[0])
        self.assertIn("sector +i of C2z", notes[0])

    def test_backend_record_under_auto(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF import soc_sectors
        from parsec_python.acceleration.cli import _soc_orbital_symmetry
        from parsec_python.acceleration.models import BackendInfo

        info = BackendInfo(requested="auto", selected="cupy", details=(("orbital_symmetry", "full grid"),))
        system, _ = _prepared([Atom("Au", [0.0, 0.0, 0.0])], iterations=1)
        view = SimpleNamespace(backend_info=info, grid=system.grid, atoms=system.atoms, input=system.input,
                               electron_count=system.electron_count)

        def text(symmetry_on=False):
            return dict(_soc_orbital_symmetry(view, "auto", symmetry_on).details)["orbital_symmetry"]

        self.assertIn("PARSEC_SOC_SECTORS=auto, the whole-grid spinor basis below the floor", text())
        self.assertIn("sector +i of C2z", text(symmetry_on=True))
        self.assertIn("(PARSEC_SOC_SECTORS=auto)", text(symmetry_on=True))
        with mock.patch.object(soc_sectors, "MINIMUM_ENTRIES", 1):
            self.assertIn("sector +i of C2z", text())


if __name__ == "__main__":
    unittest.main()
