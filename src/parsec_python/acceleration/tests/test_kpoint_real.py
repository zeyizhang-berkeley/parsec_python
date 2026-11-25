"""The Gamma point of a k-point SCF in real arithmetic (``PARSEC_KPOINT_REAL_GAMMA``, ``acceleration/SCF/kpoint_real.py``).

At ``k = 0`` the Bloch Hamiltonian is real symmetric.  The device converter gives a real operator there
(``real_arithmetic``) for the whole grid and for every sector of the little group, and nowhere else (a general k-point,
the exchange of a hybrid, the switch off, the converter of the band paths); the real operator applied to a real block
is the complex operator applied to it, and to a complex block by its parts.  The device solver of one process and the
shared columns of the MPI pools (ranks as threads) solve it in float64 with the spectrum of the dense matrix and the
same numbers to round-off; the SCF of the silicon cube at the eight time-reversal invariant k-points (one of them
Gamma) gives the energy of the complex solve, whole and in sectors, in one process and on the pools.  On a GPU the real
kernels of the sector stencil and of the whole-grid stencil give the real part of the complex kernels' result (the
sector kernel bit for bit), and the SCF of the cube the energy of the complex route.
"""

from __future__ import annotations

import os
import unittest
import warnings
from dataclasses import replace
from unittest import mock

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
from parsec_python.acceleration.SCF import kpoint_real
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.tests.test_sector_bloch_stencil import _both_routes, _helper
from parsec_python.acceleration.tests.test_kpoint_sectors import _decomposition
from parsec_python.acceleration.tests.test_kpoint_pools import _check

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


def _converter(real: bool = True, backend=None):
    converter = DeviceBlochConverter(NumpyArrayBackend() if backend is None else backend)
    converter.real_gamma = real
    return converter


def _settings():
    from parsec_python.Eigensolvers import ChebFFSettings, EigvalSettings, SubspaceSettings

    return EigvalSettings(
        safety_buffer=6,
        initial_method="chebff",
        chebff=ChebFFSettings(polynomial_degree=20, filter_cycles=3),
        subspace=SubspaceSettings(polynomial_degree=12, degree_delta=3, block_size=6),
    )


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(kpoint_real.GAMMA_SWITCH, None)
            self.assertEqual(kpoint_real.real_gamma_setting(), "auto")
        for value, expected in (("off", "off"), (" AUTO ", "auto"), ("", "auto")):
            with mock.patch.dict(os.environ, {kpoint_real.GAMMA_SWITCH: value}):
                self.assertEqual(kpoint_real.real_gamma_setting(), expected)
        with mock.patch.dict(os.environ, {kpoint_real.GAMMA_SWITCH: "on"}):
            with self.assertRaises(ValueError):
                kpoint_real.real_gamma_setting()
            # the preparation of a periodic run reads it
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches()

    def test_gamma_and_the_states(self) -> None:
        self.assertTrue(kpoint_real.is_gamma(np.zeros(3)))
        self.assertTrue(kpoint_real.is_gamma([0.0, -0.0, 0.0]))
        self.assertFalse(kpoint_real.is_gamma([0.0, 1.0e-30, 0.0]))
        self.assertFalse(kpoint_real.is_gamma(None))
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import SpinorEigvalState

        real = SpinorEigvalState(np.asfortranarray(np.ones((5, 2))), np.zeros(2), 2, 5)
        cast = kpoint_real.matched_state(real, False)
        self.assertEqual(cast.vectors.dtype, np.complex128)
        self.assertTrue(cast.vectors.flags.f_contiguous)
        np.testing.assert_array_equal(cast.vectors.real, real.vectors)
        self.assertIs(kpoint_real.matched_state(real, True), real)
        self.assertIsNone(kpoint_real.matched_state(cast, True))
        self.assertIs(kpoint_real.matched_state(cast, False), cast)
        self.assertIsNone(kpoint_real.matched_state(None, True))
        self.assertEqual(kpoint_real.as_complex(np.ones((3, 2))).dtype, np.complex128)


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _helper()

    def test_real_at_gamma_only(self) -> None:
        helper = self.helper
        side = helper.side
        full = helper._hamiltonian(np.zeros(3), "gradient")
        self.assertTrue(_converter().operator(full).real_arithmetic)
        self.assertFalse(_converter(real=False).operator(full).real_arithmetic)
        self.assertFalse(_converter().operator(full, real=False).real_arithmetic)
        for k_point in (np.pi / side * np.array([1.0, 0.0, 0.0]), np.array([0.11, 0.07, 0.05])):
            self.assertFalse(_converter().operator(helper._hamiltonian(k_point, "gradient")).real_arithmetic)
        # the Bloch form at Gamma (u gauge, no imaginary part) is real as well
        self.assertTrue(_converter().operator(helper._hamiltonian(np.zeros(3), "bloch")).real_arithmetic)

    def test_the_real_operator_applies_the_complex_one(self) -> None:
        helper = self.helper
        rng = np.random.default_rng(11)
        for kinetic in ("gradient", "bloch"):
            full = helper._hamiltonian(np.zeros(3), kinetic)
            complex_operator = _converter(real=False).operator(full)
            real_operator = _converter().operator(full)
            block = rng.normal(size=(full.shape[0], 5))
            expected = complex_operator.apply(block.astype(np.complex128))
            found = real_operator.apply(block)
            self.assertEqual(found.dtype, np.float64)
            scale = float(np.max(np.abs(expected)))
            self.assertLess(float(np.max(np.abs(expected.imag))), 1.0e-13 * scale)
            np.testing.assert_allclose(found, expected.real, rtol=0, atol=1.0e-13 * scale)
            mixed = block + 1j * rng.normal(size=block.shape)
            np.testing.assert_allclose(
                real_operator.apply(mixed), complex_operator.apply(mixed), rtol=0, atol=1.0e-13 * scale
            )

    def test_the_sectors_of_gamma(self) -> None:
        helper = self.helper
        _, _, decomposition = _decomposition(helper.cell, np.zeros(3))
        full = helper._hamiltonian(np.zeros(3), "gradient")
        rng = np.random.default_rng(12)
        found = _both_routes(np.zeros(3), decomposition, full)
        for setting in ("auto", "off"):  # the gathered stencil and the reduced CSR
            for sector, host in enumerate(found[setting]):
                real_operator = _converter().operator(host)
                self.assertTrue(real_operator.real_arithmetic, f"{setting} sector {sector}")
                complex_operator = _converter(real=False).operator(host)
                block = rng.normal(size=(host.shape[0], 3))
                expected = complex_operator.apply(block.astype(np.complex128))
                scale = float(np.max(np.abs(expected)))
                np.testing.assert_allclose(real_operator.apply(block), expected.real, rtol=0, atol=1.0e-13 * scale)
        # a sector of a general k-point stays complex
        _, _, other = _decomposition(helper.cell, np.array([0.11, 0.07, 0.05]))
        host = _both_routes(np.array([0.11, 0.07, 0.05]), other, helper._hamiltonian(np.array([0.11, 0.07, 0.05]),
                                                                                       "gradient"))["auto"][0]
        self.assertFalse(_converter().operator(host).real_arithmetic)

    def test_the_real_solve(self) -> None:
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import solve_spinor_eigval

        full = self.helper._hamiltonian(np.zeros(3), "gradient")
        operator = _converter().operator(full)
        dense = np.linalg.eigvalsh(full.apply(np.eye(full.shape[0], dtype=np.complex128)).real)
        state = None
        for _ in range(8):
            result = solve_spinor_eigval(operator, 12, settings=_settings(), state=state, xp=np)
            state = result.state
        self.assertEqual(result.vectors.dtype, np.float64)
        np.testing.assert_allclose(result.eigenvalues, dense[:12], rtol=0, atol=1.0e-6)


class ColumnSolverTests(unittest.TestCase):
    """The shared columns of the pools (ranks as threads) on the real operator of ``k = 0``: the real solve of one
    device to round-off, every rank the same numbers."""

    def _assert(self, report: dict, sizes) -> None:
        self.assertEqual(report["kinds"], ["<f8"])
        for size in sizes:
            with self.subTest(ranks=size):
                entry = report[str(size)]
                self.assertTrue(entry["ranks_equal"])
                self.assertLess(entry["value_error"], 1.0e-11)
                self.assertLess(entry["norm_error"], 1.0e-11)

    def test_rows_route(self) -> None:
        self._assert(_check("eigensolver", "chebff", "1,2,3", "4", "route=rows", "real"), (1, 2, 3))

    def test_block_route(self) -> None:
        self._assert(_check("eigensolver", "chebff", "1,2,3", "4", "route=block", "real"), (1, 2, 3))

    def test_block_route_in_many_rounds(self) -> None:
        self._assert(_check("eigensolver", "chebff", "3", "2", str(8 * 3 * 2000), "route=block", "real"), (3,))

    def test_converged_first_solve(self) -> None:
        self._assert(_check("eigensolver", "chebdav", "2", "2", "route=block", "real"), (2,))


def _silicon_cube():
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = silicon("pbe", spacing=0.855)
        base = replace(base, scf=replace(base.scf, convergence_criterion=1.0e-7))
        system = prepare_periodic_single_point(base)
    side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
    k_points = np.array([np.array(bits, dtype=float) * np.pi / side for bits in np.ndindex(2, 2, 2)])
    return system, k_points, np.full(8, 0.125)


def _floors(floor: int | None):
    """The floors of the k-point sectors, of the real Gamma point and of the psi gauge set to ``floor`` (``None``:
    the defaults), so that the small cube is split and solved in real arithmetic all the same."""
    import contextlib

    stack = contextlib.ExitStack()
    if floor is not None:
        stack.enter_context(mock.patch.object(sectors, "_MINIMUM_ENTRIES", floor))
        stack.enter_context(mock.patch.object(kpoint_real, "_MINIMUM_ENTRIES", floor))
        stack.enter_context(mock.patch.object(kpoint_real, "_PSI_MINIMUM_ENTRIES", floor))
    return stack


def _counted_scf(system, k_points, weights, environment, floor: int | None = 0):
    """The SCF of the k-points with ``environment``, and how many real and complex operators were made; ``floor``
    the floor of the k-point sectors and of the real Gamma point (``None``: the default, 2^24, which the small cube
    is below)."""
    from parsec_python.acceleration.Hamiltonian import device_bloch
    from parsec_python.SCF.kpoints import run_scf_kpoints

    counts = {"real": 0, "complex": 0}
    original = device_bloch.DeviceBlochConverter.operator
    # (the threads of this process: worker processes of the k-point sectors, PARSEC_KPOINT_SECTOR_PROCESSES=auto on
    # several GPUs, make their operators where this count does not see them; their numbers are those of the threads)
    environment = {"PARSEC_KPOINT_SECTOR_PROCESSES": "off", **environment}

    def counted(self, hamiltonian, **keywords):
        operator = original(self, hamiltonian, **keywords)
        counts["real" if operator.real_arithmetic else "complex"] += 1
        return operator

    with mock.patch.dict(os.environ, environment), mock.patch.object(
        device_bloch.DeviceBlochConverter, "operator", counted
    ), _floors(floor), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scf_kpoints(system, k_points, weights)
    return result, counts


class SCFTests(unittest.TestCase):
    """The silicon cube at the eight time-reversal invariant k-points on the device code of the host
    (``numpy-device``), whole and in sectors."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system, cls.k_points, cls.weights = _silicon_cube()

    def test_the_energy_of_the_complex_solve(self) -> None:
        for split in ("auto", "off"):
            found = {}
            for setting in ("auto", "off"):
                environment = {
                    "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1",
                    "PARSEC_KPOINT_SECTORS": split, kpoint_real.GAMMA_SWITCH: setting,
                }
                found[setting] = _counted_scf(self.system, self.k_points, self.weights, environment)
            (real, real_counts), (plain, plain_counts) = found["auto"], found["off"]
            self.assertTrue(real.converged and plain.converged)
            self.assertGreater(real_counts["real"], 0, split)
            self.assertEqual(plain_counts["real"], 0)
            # the start vectors differ (real against complex): the SCF may take a step more or less
            self.assertLessEqual(abs(real.iterations - plain.iterations), 2)
            self.assertAlmostEqual(float(real.energies.total), float(plain.energies.total), delta=1.0e-9)
            self.assertAlmostEqual(float(real.fermi_level), float(plain.fermi_level), delta=1.0e-7)
            np.testing.assert_allclose(np.asarray(real.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)

    def test_the_cpu_route_stays_complex(self) -> None:
        environment = {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_SECTORS": "off", kpoint_real.GAMMA_SWITCH: "auto"}
        from parsec_python.acceleration.Hamiltonian import device_bloch

        with mock.patch.object(device_bloch.DeviceBlochConverter, "_takes_real", side_effect=AssertionError):
            result, _counts = _counted_scf(self.system, self.k_points, self.weights, environment)
        self.assertTrue(result.converged)

    def test_a_cell_below_the_floor_keeps_the_complex_solve(self) -> None:
        # the cube (1,728 points, 22 states) is below 2^24: auto keeps the complex solve of before, bit for bit, whole
        # and split into sectors (the floor of the sectors lowered alone, as the tests of the sectors do)
        for split in (False, True):
            found = {}
            for setting in ("auto", "off"):
                environment = {
                    "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1",
                    "PARSEC_KPOINT_SECTORS": "auto", kpoint_real.GAMMA_SWITCH: setting,
                }
                with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0 if split else sectors._MINIMUM_ENTRIES):
                    found[setting] = _counted_scf(self.system, self.k_points, self.weights, environment, floor=None)
            (gated, gated_counts), (plain, _) = found["auto"], found["off"]
            self.assertEqual(gated_counts["real"], 0, split)
            self.assertEqual("kpoint_sector_tasks" in gated.timings.detail, split)
            self.assertEqual(gated.iterations, plain.iterations)
            self.assertEqual(float(gated.energies.total), float(plain.energies.total))
            np.testing.assert_array_equal(np.asarray(gated.density), np.asarray(plain.density))


class GateTests(unittest.TestCase):
    """Where the real Gamma point applies (:func:`kpoint_real.real_gamma_applies`): the switch, the floor of the
    k-point sectors on the basis of a k-point, SCF jobs only (not the band paths)."""

    @staticmethod
    def _job(states: int, geometry=None):
        from parsec_python.SCF.kpoint_solver import KPointJob

        return KPointJob(1, states, lambda *a: None, lambda *a: None, lambda *a, **k: {}, geometry=geometry)

    def test_the_floor_and_the_switch(self) -> None:
        floor = int(kpoint_real._MINIMUM_ENTRIES)
        # the number of the floor of the k-point sectors, a constant of its own: a test that splits a small cell keeps
        # the complex solve
        self.assertEqual(floor, int(sectors._MINIMUM_ENTRIES))
        self.assertEqual(floor, 1 << 24)
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0), mock.patch.dict(
            os.environ, {kpoint_real.GAMMA_SWITCH: "auto"}
        ):
            self.assertFalse(kpoint_real.real_gamma_applies(self._job(22), 1728))
        with mock.patch.dict(os.environ, {kpoint_real.GAMMA_SWITCH: "auto"}):
            self.assertTrue(kpoint_real.real_gamma_applies(self._job(64), floor // 64))
            self.assertFalse(kpoint_real.real_gamma_applies(self._job(64), floor // 64 - 1))
            self.assertFalse(kpoint_real.real_gamma_applies(self._job(64), None))
            # 64 silicon atoms at 44 points per edge with 144 states: below the floor (no gain measured there)
            self.assertFalse(kpoint_real.real_gamma_applies(self._job(144), 44 ** 3))
            # 216 carbon atoms at 54 points per edge with 456 states: above it
            self.assertTrue(kpoint_real.real_gamma_applies(self._job(456), 54 ** 3))
            # the band k-points of a Gamma cell (their invariance in the geometry): complex
            band = self._job(456, geometry={"invariance": lambda *a: None})
            self.assertFalse(kpoint_real.real_gamma_applies(band, 54 ** 3))
        with mock.patch.dict(os.environ, {kpoint_real.GAMMA_SWITCH: "off"}):
            self.assertFalse(kpoint_real.real_gamma_applies(self._job(456), 54 ** 3))

    def test_the_solvers_take_the_decision(self) -> None:
        from parsec_python.acceleration.SCF import kpoint_parallel

        original = kpoint_parallel._SolveWorker
        found: list[bool] = []

        def worker(job, tasks, index, plan=None, real_gamma=False):
            made = original(job, tasks, index, plan, real_gamma)
            found.append(bool(made.converter.real_gamma))
            return made

        environment = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1",
                       kpoint_real.GAMMA_SWITCH: "auto"}
        with mock.patch.dict(os.environ, environment), mock.patch.object(kpoint_parallel, "_SolveWorker", worker):
            for size, states, real_gamma, expected in (
                (54 ** 3, 456, None, True), (44 ** 3, 144, None, False), (44 ** 3, 144, True, True),
                (54 ** 3, 456, False, False),
            ):
                found.clear()
                solver = kpoint_parallel.PooledKPointSolver(
                    self._job(states), problem_size=size, backend="numpy-device", real_gamma=real_gamma
                )
                solver.close()
                self.assertTrue(found)
                self.assertEqual(set(found), {expected}, (size, real_gamma))

    def test_the_band_paths_keep_the_complex_solve(self) -> None:
        helper = _helper()
        side = helper.side
        k_point = np.pi / side * np.array([1.0, 0.0, 0.0])
        full = helper._hamiltonian(k_point, "bloch")
        geometry = {"grid": helper.system.grid}
        with mock.patch.dict(os.environ, {kpoint_real.TRIM_SWITCH: "auto"}):
            # the small cube is below the floor of the psi gauge (2^22 entries)
            self.assertEqual(int(kpoint_real._PSI_MINIMUM_ENTRIES), 1 << 22)
            self.assertIsNone(kpoint_real.psi_gauge_for(self._job(8, geometry), 0, full, {}))
            with _floors(0):
                gauge = kpoint_real.psi_gauge_for(self._job(8, geometry), 0, full, {})
                self.assertIsInstance(gauge, kpoint_real.PsiGauge)
                band = self._job(8, dict(geometry, invariance=lambda *a: None))
                self.assertIsNone(kpoint_real.psi_gauge_for(band, 0, full, {}))


class BufferTests(unittest.TestCase):
    """The round buffers of the block route (ranks as threads): one allocation per name for the complex k-points and
    the real Gamma point of a pool, the bytes of the complex solve."""

    def test_one_allocation_for_both_dtypes(self) -> None:
        # the whole share in one round (the default exchange size), then rounds of 200 kB
        for exchange in (None, "200000"):
            report = _check("buffers", "2") if exchange is None else _check("buffers", "2", exchange)
            for found in report["ranks"]:
                complex_first, real_first = found["complex-real"], found["real-complex"]
                complex_alone = complex_first[0]
                with self.subTest(exchange=exchange):
                    self.assertEqual(set(complex_alone), {"send", "receive", "applied"})
                    # the real solve after the complex one takes the same allocations
                    self.assertEqual(complex_first[1], complex_alone)
                    # a real solve first takes at most the bytes of the complex solve (half of them where the share
                    # is one round), the complex one after it grows them to the complex solve's
                    self.assertEqual(set(real_first[0]), set(complex_alone))
                    for name, size in real_first[0].items():
                        limit = complex_alone[name] if exchange else complex_alone[name] // 2 + 8
                        self.assertLessEqual(size, limit, name)
                    self.assertEqual(real_first[1], complex_alone)


class TrimTests(unittest.TestCase):
    """The time-reversal invariant points in the Bloch form: the conjugate pairs of their sectors
    (``PARSEC_KPOINT_TRIM_REAL``) and the kinetic form ``trim`` (``PARSEC_KPOINT_KINETIC``)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _helper()

    def test_the_switches(self) -> None:
        from parsec_python.Laplacian.finite_difference import kpoint_kinetic_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(kpoint_real.TRIM_SWITCH, None)
            self.assertEqual(kpoint_real.trim_real_setting(), "auto")
        with mock.patch.dict(os.environ, {kpoint_real.TRIM_SWITCH: "OFF"}):
            self.assertEqual(kpoint_real.trim_real_setting(), "off")
        for name, value in ((kpoint_real.TRIM_SWITCH, "on"), ("PARSEC_KPOINT_KINETIC", "psi")):
            with mock.patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    sectors.check_kpoint_sector_switches()
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "trim"}):
            self.assertEqual(kpoint_kinetic_setting(), "trim")

    def test_the_trim_kinetic_form(self) -> None:
        from parsec_python.Laplacian.finite_difference import bloch_form_at, bloch_laplacian_or_none, is_trim_point

        grid = self.helper.system.grid
        side = self.helper.side
        trim, quarter = np.pi / side * np.array([1.0, 0.0, 1.0]), np.pi / side * np.array([0.5, 0.0, 0.0])
        self.assertTrue(is_trim_point(grid, trim))
        self.assertFalse(is_trim_point(grid, quarter))
        self.assertFalse(is_trim_point(grid, np.zeros(3)))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "trim"}):
            self.assertIsNotNone(bloch_laplacian_or_none(grid, trim))
            self.assertTrue(bloch_form_at(grid, trim))
            for k_point in (quarter, np.zeros(3)):
                self.assertIsNone(bloch_laplacian_or_none(grid, k_point))
                self.assertFalse(bloch_form_at(grid, k_point))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "gradient"}):
            self.assertFalse(bloch_form_at(grid, trim))

    def test_the_partners_have_one_spectrum(self) -> None:
        from parsec_python.acceleration.tests.test_kpoint_sectors import _dense

        helper = self.helper
        side = helper.side
        for bits in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)):
            k_point = np.pi / side * np.array(bits)
            members = sectors.generated_subgroup(helper.cell.table, helper.cell.little_group(k_point, lambda index: True))
            elements, generators = sectors.largest_commuting_involutions(helper.cell.table, members)
            partners = kpoint_real.trim_partners(helper.cell, elements, generators, k_point)
            self.assertIsNotNone(partners, bits)
            order = 1 << len(generators)
            np.testing.assert_array_equal(partners[partners], np.arange(order))
            self.assertFalse(np.any(partners == np.arange(order)))
            _, _, decomposition = _decomposition(helper.cell, k_point)
            full = helper._hamiltonian(k_point, "bloch")
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * order)
            for sector in range(order):
                other = int(partners[sector])
                if other < sector:
                    continue
                self.assertEqual(decomposition.sector_sizes[sector], decomposition.sector_sizes[other])
                mine = np.linalg.eigvalsh(_dense(split.hamiltonian(sector, full)))
                theirs = np.linalg.eigvalsh(_dense(split.hamiltonian(other, full)))
                np.testing.assert_allclose(mine, theirs, rtol=0, atol=1.0e-11)
        # the gradient form is not a reason for partners, and Gamma has none
        self.assertIsNone(kpoint_real.trim_partners(helper.cell, [0], [], np.zeros(3)))

    def test_the_scf_with_the_pairs(self) -> None:
        system, k_points, weights = _silicon_cube()
        found = {}
        for kinetic, setting in (("bloch", "auto"), ("bloch", "off"), ("trim", "auto")):
            environment = {
                "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1", "PARSEC_KPOINT_SECTORS": "auto",
                "PARSEC_KPOINT_KINETIC": kinetic, kpoint_real.TRIM_SWITCH: setting,
            }
            found[(kinetic, setting)] = _counted_scf(system, k_points, weights, environment)[0]
        paired, plain, trim = found[("bloch", "auto")], found[("bloch", "off")], found[("trim", "auto")]
        self.assertGreater(paired.timings.detail.get("kpoint_sector_pairs", 0.0), 0.0)
        self.assertNotIn("kpoint_sector_pairs", plain.timings.detail)
        self.assertLess(paired.timings.detail["kpoint_sector_work_fraction"],
                        plain.timings.detail["kpoint_sector_work_fraction"])
        for result in (paired, trim):
            self.assertTrue(result.converged)
            self.assertLessEqual(abs(result.iterations - plain.iterations), 2)
            self.assertAlmostEqual(float(result.energies.total), float(plain.energies.total), delta=1.0e-9)
            np.testing.assert_allclose(np.asarray(result.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)

    def test_the_forces_with_the_pairs(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
        problem = replace(
            base,
            scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
            eigensolver=replace(base.eigensolver, tolerance=1.0e-10),
        )
        # the sublattice of atom 0 moved along x: the face-centring translations stay (the partners of X), and the
        # two sublattices push each other (forces)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        first = np.asarray(problem.atoms[0].position, dtype=float)
        atoms = []
        for atom in problem.atoms:
            position = np.asarray(atom.position, dtype=float)
            steps = (position - first) / (0.5 * side)
            if np.allclose(steps, np.round(steps), atol=1.0e-8) and int(np.round(steps).sum()) % 2 == 0:
                position = position + np.array([0.1, 0.0, 0.0])
            atoms.append(replace(atom, position=position))
        problem = replace(problem, atoms=atoms)
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        weights = np.array([0.5, 0.5])
        # in sectors (the conjugate pairs) and whole (the psi gauge of X)
        for split in ("auto", "off"):
            results = {}
            for setting in ("auto", "off"):
                switches = {
                    "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1",
                    "PARSEC_KPOINT_SECTORS": split, "PARSEC_KPOINT_KINETIC": "bloch", kpoint_real.TRIM_SWITCH: setting,
                }
                with mock.patch.dict(os.environ, switches), _floors(0), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    system = prepare_periodic_single_point(problem)
                    results[setting] = run_scf_kpoints(system, k_points, weights, compute_forces=True)
            paired, plain = results["auto"], results["off"]
            self.assertTrue(paired.converged and plain.converged)
            if split == "auto":
                self.assertGreater(paired.timings.detail.get("kpoint_sector_pairs", 0.0), 0.0)
            self.assertGreater(float(np.max(np.abs(plain.nonlocal_forces))), 1.0e-3)
            self.assertAlmostEqual(paired.energies.total, plain.energies.total, delta=1.0e-9)
            np.testing.assert_allclose(paired.nonlocal_forces, plain.nonlocal_forces, rtol=0.0, atol=1.0e-8)

    def test_the_psi_gauge(self) -> None:
        helper = self.helper
        side = helper.side
        n = helper.system.grid.size
        for bits in ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
            k_point = np.pi / side * np.array(bits)
            full = helper._hamiltonian(k_point, "bloch")
            gauge = kpoint_real.PsiGauge(helper.system.grid, k_point)
            psi = gauge.hamiltonian(full)
            self.assertEqual(psi.bloch_laplacian.dtype, np.float64)
            self.assertEqual(psi.nonlocal_operator.projectors.dtype, np.float64)
            operator = _converter(real=False).operator(psi)  # real without PARSEC_KPOINT_REAL_GAMMA
            self.assertTrue(operator.real_arithmetic)
            dense_psi = operator.apply(np.eye(n))
            dense_u = full.apply(np.eye(n, dtype=np.complex128))
            phases = gauge.phases
            np.testing.assert_allclose(
                dense_psi, (phases[:, None] * dense_u * np.conj(phases)[None, :]).real, rtol=0, atol=1.0e-12
            )
            block = np.random.default_rng(3).normal(size=(n, 2))
            np.testing.assert_allclose(gauge.to_u(block), np.conj(phases)[:, None] * block, rtol=0, atol=0)

    def test_the_unsplit_points_in_the_psi_gauge(self) -> None:
        system, k_points, weights = _silicon_cube()
        found = {}
        for setting in ("auto", "off"):
            environment = {
                "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1", "PARSEC_KPOINT_SECTORS": "off",
                "PARSEC_KPOINT_KINETIC": "bloch", kpoint_real.TRIM_SWITCH: setting,
            }
            found[setting] = _counted_scf(system, k_points, weights, environment)
        (psi, psi_counts), (plain, plain_counts) = found["auto"], found["off"]
        # seven time-reversal invariant points other than Gamma, and Gamma, real in every step
        self.assertEqual(psi_counts["complex"], 0)
        self.assertGreater(plain_counts["complex"], 0)
        self.assertTrue(psi.converged and plain.converged)
        self.assertLessEqual(abs(psi.iterations - plain.iterations), 2)
        self.assertAlmostEqual(float(psi.energies.total), float(plain.energies.total), delta=1.0e-9)
        np.testing.assert_allclose(np.asarray(psi.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)

    def test_the_pairs_on_pools(self) -> None:
        for ranks, pools in (("3", "3"), ("2", "1")):
            report = _check("sectors", ranks, pools, "kinetic=bloch")
            serial = report["serial"]
            self.assertGreater(serial["pairs"], 0.0)
            for result in report["pooled"]:
                self.assertTrue(result["converged"])
                self.assertEqual(result["pairs"], serial["pairs"])
                self.assertEqual(result["iterations"], serial["iterations"])
                self.assertAlmostEqual(result["energy"], serial["energy"], delta=1.0e-8)

    def test_the_real_gamma_point_on_pools(self) -> None:
        # the cube in sectors on pools of one rank and on one pool of two ranks (shared columns): the real Gamma point
        # (its floor at 0) against the complex solve (the small cube below the floor)
        for ranks, pools in (("3", "3"), ("2", "1")):
            real = _check("sectors", ranks, pools, "real")
            plain = _check("sectors", ranks, pools)
            self.assertGreater(real["serial"]["real_operators"], 0)
            self.assertGreater(real["pooled_real_operators"], 0)
            self.assertEqual(plain["serial"]["real_operators"], 0)
            self.assertEqual(plain["pooled_real_operators"], 0)
            for result in real["pooled"] + [real["serial"]]:
                self.assertTrue(result["converged"])
                self.assertAlmostEqual(result["energy"], plain["serial"]["energy"], delta=1.0e-8)

    def test_the_psi_gauge_on_pools(self) -> None:
        # one task per k-point: the psi gauge of the seven points other than Gamma on pools of one rank and on one
        # pool of two ranks (shared columns), against the u gauge of before
        for ranks, pools in (("3", "3"), ("2", "1")):
            psi = _check("sectors", ranks, pools, "kinetic=bloch", "whole", "real")
            plain = _check("sectors", ranks, pools, "kinetic=bloch", "whole", "trimoff")
            self.assertGreater(psi["pooled_real_operators"], 0)
            self.assertEqual(plain["pooled_real_operators"], 0)
            self.assertEqual(psi["serial"]["tasks"], 0.0)  # no sectors
            for result in psi["pooled"] + [psi["serial"]]:
                self.assertTrue(result["converged"])
                self.assertAlmostEqual(result["energy"], plain["serial"]["energy"], delta=1.0e-8)
            for result in psi["pooled"]:
                # the shared columns factor by Cholesky, the one-process solver by Householder: a step more or less
                self.assertLessEqual(abs(result["iterations"] - psi["serial"]["iterations"]), 2)


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _helper()

    def test_the_real_kernel_of_a_sector(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian import sector_bloch_stencil as sbs
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        helper = self.helper
        _, _, decomposition = _decomposition(helper.cell, np.zeros(3))
        full = helper._hamiltonian(np.zeros(3), "gradient")
        rng = np.random.default_rng(13)
        for host in _both_routes(np.zeros(3), decomposition, full)["auto"]:
            stencil = sbs.DeviceSectorStencil(cp, host.bloch_laplacian)
            self.assertTrue(stencil.real_capable)
            block = cp.asarray(rng.normal(size=(host.shape[0], 37)))
            previous = cp.asarray(rng.normal(size=(host.shape[0], 37)))
            added = cp.asarray(rng.normal(size=(host.shape[0], 37)))
            before = cp.asarray(rng.uniform(0.1, 0.9, 37))
            after = cp.asarray(rng.uniform(0.5, 1.5, 37))
            diagonal = stencil.diagonal(cp.asarray(np.ascontiguousarray(host._diagonal)))
            for keywords in ({}, {"added": added, "previous": previous, "center": 1.7, "scale": 0.3, "before": before,
                                  "after": after}):
                real = stencil.apply(block, diagonal, **keywords)
                self.assertEqual(real.dtype, cp.float64)
                complex_keywords = {
                    name: (value.astype(cp.complex128) if name in ("added", "previous") else value)
                    for name, value in keywords.items()
                }
                reference = stencil.apply(block.astype(cp.complex128), diagonal, **complex_keywords)
                # the real part of the complex kernel's result, bit for bit
                np.testing.assert_array_equal(cp.asnumpy(real), cp.asnumpy(reference.real))
            converter = _converter(backend=CupyArrayBackend(0))
            operator = converter.operator(host)
            self.assertTrue(operator.real_arithmetic and operator.fused_sector_recurrence)
            expected = _converter(real=False, backend=CupyArrayBackend(0)).operator(host).apply(
                block.astype(cp.complex128)
            )
            scale = float(cp.max(cp.abs(expected)))
            np.testing.assert_allclose(cp.asnumpy(operator.apply(block)), cp.asnumpy(expected.real), rtol=0,
                                       atol=1.0e-13 * scale)

    def test_the_real_kernel_of_the_whole_grid(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        full = self.helper._hamiltonian(np.zeros(3), "gradient")
        rng = np.random.default_rng(14)
        real_converter = _converter(backend=CupyArrayBackend(0))
        real_converter.fused_stencil = True
        complex_converter = _converter(real=False, backend=CupyArrayBackend(0))
        complex_converter.fused_stencil = True
        real_operator, complex_operator = real_converter.operator(full), complex_converter.operator(full)
        self.assertTrue(real_operator.real_arithmetic)
        self.assertIsNotNone(real_operator.stencil)
        self.assertTrue(real_operator.stencil.real_capable)
        block = cp.asarray(rng.normal(size=(full.shape[0], 33)))
        previous = cp.asarray(rng.normal(size=(full.shape[0], 33)))
        before = cp.asarray(rng.uniform(0.1, 0.9, 33))
        after = cp.asarray(rng.uniform(0.5, 1.5, 33))
        expected = complex_operator.apply(block.astype(cp.complex128))
        scale = float(cp.max(cp.abs(expected)))
        np.testing.assert_allclose(cp.asnumpy(real_operator.apply(block)), cp.asnumpy(expected.real), rtol=0,
                                   atol=1.0e-13 * scale)
        step = real_operator.chebyshev_step(block, previous, 1.7, 0.3, before, after)
        reference = complex_operator.chebyshev_step(
            block.astype(cp.complex128), previous.astype(cp.complex128), 1.7, 0.3, before, after
        )
        self.assertEqual(step.dtype, cp.float64)
        np.testing.assert_allclose(cp.asnumpy(step), cp.asnumpy(reference.real), rtol=0,
                                   atol=1.0e-13 * float(cp.max(cp.abs(reference))))

    def test_the_scf_on_the_gpu(self) -> None:
        system, k_points, weights = _silicon_cube()
        for split in ("auto", "off"):
            found = {}
            for setting in ("auto", "off"):
                environment = {
                    "PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_KPOINT_SECTORS": split, kpoint_real.GAMMA_SWITCH: setting,
                }
                found[setting] = _counted_scf(system, k_points, weights, environment)
            (real, real_counts), (plain, plain_counts) = found["auto"], found["off"]
            self.assertTrue(real.converged and plain.converged)
            self.assertGreater(real_counts["real"], 0, split)
            self.assertEqual(plain_counts["real"], 0)
            self.assertAlmostEqual(float(real.energies.total), float(plain.energies.total), delta=1.0e-9)
            np.testing.assert_allclose(np.asarray(real.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)

    def test_the_twisted_kernel_of_the_psi_gauge(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        helper = self.helper
        side = helper.side
        rng = np.random.default_rng(15)
        for bits in ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)):
            k_point = np.pi / side * np.array(bits)
            gauge = kpoint_real.PsiGauge(helper.system.grid, k_point)
            psi = gauge.hamiltonian(helper._hamiltonian(k_point, "bloch"))
            converter = _converter(real=False, backend=CupyArrayBackend(0))
            converter.fused_stencil = True
            operator = converter.operator(psi)
            self.assertTrue(operator.real_arithmetic)
            self.assertIsNotNone(operator.stencil, bits)
            self.assertEqual(operator.stencil.twist, tuple(-1.0 if bit else 1.0 for bit in bits))
            sparse = _converter(real=False, backend=CupyArrayBackend(0)).operator(psi)  # the real sparse route
            self.assertTrue(sparse.real_arithmetic)
            self.assertIsNone(sparse.stencil)
            block = cp.asarray(rng.normal(size=(psi.shape[0], 21)))
            expected = sparse.apply(block)
            scale = float(cp.max(cp.abs(expected)))
            np.testing.assert_allclose(cp.asnumpy(operator.apply(block)), cp.asnumpy(expected), rtol=0, atol=1.0e-13 * scale)
            previous = cp.asarray(rng.normal(size=block.shape))
            before, after = cp.asarray(rng.uniform(0.1, 0.9, 21)), cp.asarray(rng.uniform(0.5, 1.5, 21))
            step = operator.chebyshev_step(block, previous, 1.7, 0.3, before, after)
            reference = ((expected - 1.7 * block) * 0.3 - before[None, :] * previous) * after[None, :]
            np.testing.assert_allclose(cp.asnumpy(step), cp.asnumpy(reference), rtol=0,
                                       atol=1.0e-13 * float(cp.max(cp.abs(reference))))

    def test_the_psi_gauge_on_the_gpu(self) -> None:
        system, k_points, weights = _silicon_cube()
        found = {}
        for setting in ("auto", "off"):
            environment = {
                "PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_KPOINT_SECTORS": "off", "PARSEC_KPOINT_KINETIC": "bloch",
                kpoint_real.TRIM_SWITCH: setting,
            }
            found[setting] = _counted_scf(system, k_points, weights, environment)
        (psi, psi_counts), (plain, _) = found["auto"], found["off"]
        self.assertEqual(psi_counts["complex"], 0)
        self.assertAlmostEqual(float(psi.energies.total), float(plain.energies.total), delta=1.0e-9)
        np.testing.assert_allclose(np.asarray(psi.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)

    def test_the_band_path_through_gamma_stays_complex(self) -> None:
        # a Gamma cell in sectors and its band path through Gamma in the sectors of the little groups
        # (PARSEC_BAND_KPOINT_SECTORS=auto): the band k-points keep their complex solves
        import tempfile
        from pathlib import Path

        from parsec_python.acceleration.Hamiltonian import device_bloch
        from parsec_python.acceleration.tests.test_band_kpoint_sectors import _bands, _cupy_ready

        if not _cupy_ready():
            self.skipTest("the native extension is unavailable")
        counts = {"real": 0, "complex": 0}
        original = device_bloch.DeviceBlochConverter.operator

        def counted(converter, hamiltonian, **keywords):
            operator = original(converter, hamiltonian, **keywords)
            counts["real" if operator.real_arithmetic else "complex"] += 1
            return operator

        # the floors of the real route at 0: the band jobs alone keep the complex solve
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name, mock.patch.dict(
            os.environ, {kpoint_real.GAMMA_SWITCH: "auto"}
        ), mock.patch.object(device_bloch.DeviceBlochConverter, "operator", counted), mock.patch.object(
            kpoint_real, "_MINIMUM_ENTRIES", 0
        ), mock.patch.object(kpoint_real, "_PSI_MINIMUM_ENTRIES", 0):
            status, bands, calls, errors = _bands(Path(name) / "sectors", "auto")
        self.assertEqual(status, 0, errors[-4000:])
        self.assertEqual(calls, [True])
        self.assertGreater(counts["complex"], 0)
        self.assertEqual(counts["real"], 0)

    def test_the_pairs_on_the_gpu(self) -> None:
        system, k_points, weights = _silicon_cube()
        found = {}
        for setting in ("auto", "off"):
            environment = {
                "PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_KINETIC": "bloch",
                kpoint_real.TRIM_SWITCH: setting,
            }
            found[setting] = _counted_scf(system, k_points, weights, environment)[0]
        paired, plain = found["auto"], found["off"]
        self.assertGreater(paired.timings.detail.get("kpoint_sector_pairs", 0.0), 0.0)
        self.assertAlmostEqual(float(paired.energies.total), float(plain.energies.total), delta=1.0e-9)
        np.testing.assert_allclose(np.asarray(paired.density), np.asarray(plain.density), rtol=0, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
