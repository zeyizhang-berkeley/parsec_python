"""Unpolarized Gamma-point hybrids of bulk cells in the k-point sectors (``acceleration/SCF/gamma_hybrid_route.py``).

The route takes a hybrid of a bulk orthorhombic cell at the Gamma point, unpolarized, as a single point whose basis
reaches the floor of the k-point sectors and whose little group at least halves the eigensolver work; every other
input keeps the device Fock loop of the full grid.  The reference solver runs such an input as the 1x1x1 k-point
mesh in the sectors of the little group of Gamma, with the total energy of the real Gamma loop (on the host here,
on the device on a GPU).
"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python import cli as reference_cli
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import cli as accelerated_cli
from parsec_python.acceleration.SCF import gamma_hybrid_route as route_module
from parsec_python.acceleration.SCF import kpoint_sectors

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
    from parsec_python.acceleration.backends.native import native_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

    def native_available() -> bool:
        return False

_ROOT = Path(__file__).resolve().parents[4]
_SILICON = _ROOT / "examples" / "3d_Si"


def _write_silicon(folder: Path, *, functional: str = "pbe0", spacing: str = "0.855", extra=()) -> Path:
    """The conventional Si cube of ``examples/3d_Si`` (8 atoms) with ``functional`` at the Gamma point, tight."""

    changes = {
        "Grid_Spacing": spacing,
        "Expansion_Order": "8",
        "States_Num": "22",
        "Convergence_Criterion": "1.0e-9",
        "Diag_Tolerance": "1.0e-10",
        "Fermi_Temp": "1",
        "Correlation_Type": functional,
    }
    lines = []
    for line in (_SILICON / "parsec.in").read_text().splitlines():
        keyword = line.split(":", 1)[0].strip()
        lines.append(f" {keyword}: {changes[keyword]}" if ":" in line and keyword in changes else line)
    lines += [" Exx_Tolerance: 1e-10 Ry", *extra]
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(_SILICON / "Si_POTRE.DAT", folder)
    path = folder / "parsec.in"
    path.write_text("\n".join(lines) + "\n")
    return path


def _translation(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


def _final_energy(path: Path) -> float:
    found = re.findall(r"Total Energy \(PBE0\)\s+=\s+(-?[0-9.]+)", path.read_text())
    return float(found[-1])


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("AUTO", "auto"), ("off", "off"), ("0", "off"), ("kpoint", "kpoint")):
            with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": value}):
                self.assertEqual(route_module.gamma_hybrid_sectors_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "on"}):
            with self.assertRaises(ValueError):
                route_module.gamma_hybrid_sectors_setting()


class RouteDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_hybrid_"))
        cls.addClassCleanup(shutil.rmtree, cls.folder, True)
        cls.pbe0 = _translation(_write_silicon(cls.folder / "pbe0"))

    def _route(self, translation, **environment):
        environment = {"PARSEC_GAMMA_HYBRID_SECTORS": "kpoint", **environment}
        with mock.patch.dict(os.environ, environment), mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0):
            return route_module.gamma_hybrid_sector_route(translation)

    def test_the_silicon_cube_splits_into_the_sixteen_sectors_of_gamma(self) -> None:
        found = self._route(self.pbe0)
        self.assertIsNotNone(found)
        self.assertEqual(found.sectors, 16)
        self.assertEqual(found.points_per_axis, (12, 12, 12))
        self.assertEqual(found.states, 22)
        self.assertLessEqual(found.work_fraction, 0.5)
        self.assertIn("PARSEC_GAMMA_HYBRID_SECTORS=kpoint", found.note())

    def test_below_the_floor_of_the_kpoint_sectors_the_full_grid_stays(self) -> None:
        # 1728 rows x 22 states: far below 2^24
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "kpoint"}):
            self.assertIsNone(route_module.gamma_hybrid_sector_route(self.pbe0))

    def test_the_switches_keep_the_accelerated_driver(self) -> None:
        # auto (the real sectors of the Gamma route) and off (the full grid) keep the input in the driver
        self.assertIsNone(self._route(self.pbe0, PARSEC_GAMMA_HYBRID_SECTORS="off"))
        self.assertIsNone(self._route(self.pbe0, PARSEC_GAMMA_HYBRID_SECTORS="auto"))
        self.assertIsNone(self._route(self.pbe0, PARSEC_KPOINT_SECTORS="off"))

    def test_inputs_the_route_does_not_take(self) -> None:
        problem = self.pbe0.problem
        scf = problem.scf
        refused = {
            "semilocal": replace(self.pbe0, problem=replace(problem, scf=replace(scf, xc_functional="pbe", exact_exchange=None))),
            "spin": replace(self.pbe0, problem=replace(problem, scf=replace(scf, spin_polarized=True))),
            "forces": replace(self.pbe0, compute_forces=True),
            "restart": replace(self.pbe0, restart_run=True),
            "refinement": replace(self.pbe0, energy_refinement="pbe0"),
        }
        for label, translation in refused.items():
            self.assertIsNone(self._route(translation), label)
        # a Gamma-centred mesh beyond Gamma
        meshed = _translation(
            _write_silicon(
                self.folder / "mesh",
                extra=(" Kpoint_Method: mp", " begin Monkhorst_Pack_Grid", " 2 2 2", " end Monkhorst_Pack_Grid"),
            )
        )
        self.assertIsNone(self._route(meshed))

    def test_a_cell_without_symmetry_keeps_the_full_grid(self) -> None:
        atoms = list(self.pbe0.problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.13, 0.07, 0.03]))
        moved = replace(self.pbe0, problem=replace(self.pbe0.problem, atoms=atoms))
        self.assertIsNone(self._route(moved))

    def test_the_gamma_route_takes_a_hybrid_where_its_caller_solves_it_in_sectors(self) -> None:
        from parsec_python.acceleration.driver import periodic_sector_refusal

        problem = self.pbe0.problem
        self.assertEqual(periodic_sector_refusal(problem), "a hybrid functional")
        self.assertIsNone(periodic_sector_refusal(problem, hybrid_sectors=True))
        spin = replace(problem, scf=replace(problem.scf, spin_polarized=True))
        self.assertIsNotNone(periodic_sector_refusal(spin, hybrid_sectors=True))

    def test_the_reference_solver_asks_for_the_device_workers(self) -> None:
        with mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0), mock.patch.dict(
            os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "kpoint"}
        ):
            with mock.patch.object(reference_cli, "parallel_backend_uses_device", return_value=True):
                self.assertTrue(reference_cli._gamma_hybrid_on_kpoint_driver(self.pbe0, True))
                self.assertFalse(reference_cli._gamma_hybrid_on_kpoint_driver(self.pbe0, False))
                with mock.patch.dict(os.environ, {"PARSEC_MPI": "1"}):
                    self.assertFalse(reference_cli._gamma_hybrid_on_kpoint_driver(self.pbe0, True))
            with mock.patch.object(reference_cli, "parallel_backend_uses_device", return_value=False):
                self.assertFalse(reference_cli._gamma_hybrid_on_kpoint_driver(self.pbe0, True))

    def test_the_accelerated_command_line_hands_the_input_over(self) -> None:
        arguments = SimpleNamespace(quiet=False, mpi_rank=0)
        found = route_module.GammaHybridSectorRoute((12, 12, 12), 22, 16, 8, 0.1)
        with mock.patch.object(route_module, "gamma_hybrid_sector_route", return_value=found), mock.patch.object(
            accelerated_cli, "reference_main", return_value=7
        ) as reference, mock.patch.object(accelerated_cli, "_reference_argv", return_value=["x"]):
            errors = StringIO()
            with redirect_stderr(errors):
                self.assertEqual(accelerated_cli._gamma_hybrid_sector_handover(arguments, self.pbe0), 7)
            reference.assert_called_once_with(["x"])
            self.assertIn("16 sectors", errors.getvalue())
        with mock.patch.object(route_module, "gamma_hybrid_sector_route", return_value=None), mock.patch.object(
            accelerated_cli, "reference_main"
        ) as reference:
            self.assertIsNone(accelerated_cli._gamma_hybrid_sector_handover(arguments, self.pbe0))
            reference.assert_not_called()
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "on"}), redirect_stderr(StringIO()):
            self.assertEqual(accelerated_cli._gamma_hybrid_sector_handover(arguments, self.pbe0), 2)


class ReferenceRouteTests(unittest.TestCase):
    """The reference command line on the host (the device code of the k-point workers on the host, a test double)."""

    def test_the_sectors_of_gamma_give_the_energy_of_the_real_loop(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_hybrid_cli_"))
        self.addCleanup(shutil.rmtree, folder, True)
        energies, logs = {}, {}
        for setting in ("off", "kpoint"):
            path = _write_silicon(folder / setting)
            environment = {"PARSEC_GAMMA_HYBRID_SECTORS": setting, "PARSEC_KPOINT_BACKEND": "numpy-device"}
            with mock.patch.dict(os.environ, environment), mock.patch.object(
                kpoint_sectors, "_MINIMUM_ENTRIES", 0
            ), redirect_stdout(StringIO()), redirect_stderr(StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(reference_cli.main([str(path), "--no-archive", "--quiet"]), 0, setting)
            energies[setting] = _final_energy(path.parent / "parsec.out")
            logs[setting] = (path.parent / "parsec.out").read_text()
        self.assertNotIn("K-point sectors", logs["off"])
        self.assertRegex(logs["kpoint"], r"K-point sectors\s+:\s+16\s+tasks")
        # the outputs print 8 decimals
        self.assertAlmostEqual(energies["kpoint"], energies["off"], delta=2.0e-8)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class DeviceRouteTests(unittest.TestCase):
    """main.py on a GPU: the device Fock loop of the full grid (off) against the real sectors of the Gamma route
    (auto) and the 1x1x1 k-point route in the sectors of the little group of Gamma (kpoint)."""

    def test_the_sectors_on_the_gpu_give_the_energy_of_the_full_grid(self) -> None:
        from contextlib import ExitStack

        from parsec_python.acceleration.Hamiltonian import exchange_device

        folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_hybrid_gpu_"))
        self.addCleanup(shutil.rmtree, folder, True)
        energies, logs, notes = {}, {}, {}
        real_fits = exchange_device.pooled_exchange_fits
        checks = []

        def fits_once(*args, **kwargs):
            # the free memory of the device drops after the first build (mocked: it then takes no pooled block)
            checks.append(1)
            if len(checks) == 1:
                return real_fits(*args, **kwargs)
            with mock.patch.object(exchange_device, "_device_memory", return_value=(0, 1)):
                return real_fits(*args, **kwargs)

        # auto_tiles: the sector blocks paired tile by tile, without the pooled full-grid block; auto_full: auto on
        # a device whose free memory (mocked) takes no pooled block, so every build falls back to those tiles;
        # auto_late: the first build pooled, every later one on those tiles
        for setting in ("off", "auto", "auto_tiles", "auto_full", "auto_late", "kpoint"):
            path = _write_silicon(folder / setting, spacing="0.57")
            errors = StringIO()
            environment = {
                "PARSEC_GAMMA_HYBRID_SECTORS": setting.split("_")[0],
                "PARSEC_CUPY_EXCHANGE_POOL": "off" if setting == "auto_tiles" else "auto",
            }
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, environment))
                stack.enter_context(mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0))
                if setting == "auto_full":
                    stack.enter_context(mock.patch.object(exchange_device, "_device_memory", return_value=(0, 1)))
                if setting == "auto_late":
                    stack.enter_context(mock.patch.object(exchange_device, "pooled_exchange_fits", fits_once))
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(redirect_stderr(errors))
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
                self.assertEqual(accelerated_cli.main([str(path), "--no-archive", "--gpus", "1"]), 0, setting)
            energies[setting] = _final_energy(path.parent / "parsec.out")
            logs[setting] = (path.parent / "parsec.out").read_text()
            notes[setting] = errors.getvalue()
        # the fallback is the path of the switch off, to the printed digits, and says so
        self.assertEqual(energies["auto_full"], energies["auto_tiles"])
        self.assertIn("full-grid block of the orbitals did not fit the device", logs["auto_full"])
        self.assertNotIn("did not fit the device", logs["auto"])
        # each Fock iteration names the builds of its step, the summary those of the run
        summary = r"Pair-density Poisson solver: .* in (\d+) of (\d+) builds, where the full-grid block"
        found = re.search(summary, logs["auto_full"])
        self.assertIsNotNone(found)
        self.assertEqual(found.group(1), found.group(2))
        steps = re.findall(r"^   Exact exchange: \d+ pair-density Poisson solves, .*$", logs["auto_late"], re.M)
        self.assertGreater(len(steps), 2)
        self.assertGreater(len(checks), 2)
        self.assertNotIn("did not fit the device", steps[0])
        for line in steps[1:]:
            self.assertRegex(line, r" in (\d+) of \1 builds, where the full-grid block of the orbitals did not fit")
        found = re.search(summary, logs["auto_late"])
        self.assertIsNotNone(found)
        self.assertEqual(int(found.group(1)), int(found.group(2)) - 1)
        self.assertAlmostEqual(energies["auto_late"], energies["auto"], delta=3.0e-8)
        self.assertIn("does not take a hybrid functional; the full grid is used", logs["off"])
        # the 16 commuting involutions of the cube with its face-centring translations
        self.assertIn("exact commuting group of 16 operations detected", logs["auto"])
        self.assertNotIn("Gamma-point cell", notes["auto"])
        self.assertIn("NOTE: the hybrid functional of this Gamma-point cell", notes["kpoint"])
        self.assertRegex(logs["kpoint"], r"K-point sectors\s+:\s+16\s+tasks")
        # the outputs print 8 decimals; the three loops converge to 1e-9 Ry (SCF) and 1e-10 Ry (Fock loop)
        self.assertAlmostEqual(energies["auto"], energies["off"], delta=3.0e-8)
        self.assertAlmostEqual(energies["auto_tiles"], energies["off"], delta=3.0e-8)
        self.assertAlmostEqual(energies["kpoint"], energies["off"], delta=3.0e-8)


class ExchangePoolTests(unittest.TestCase):
    """Which sector orbitals the device exchange pools into one full-grid block (``PARSEC_CUPY_EXCHANGE_POOL``)."""

    def test_the_switch_and_the_orbitals_it_takes(self) -> None:
        from parsec_python.acceleration.Hamiltonian import exchange_device

        periodic = SimpleNamespace(small_build=lambda *args, **kwargs: False, parity_layout=lambda: None)
        cluster = SimpleNamespace(parity_layout=lambda: None)  # the Kronecker solver: no small_build
        expansion = object()

        def orbitals(*expansions):
            return SimpleNamespace(sectors=tuple(SimpleNamespace(expansion=item) for item in expansions))

        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": ""}):
            self.assertEqual(exchange_device.exchange_pool_setting(), "auto")
            self.assertTrue(exchange_device.pooled_exchange_applies(orbitals(expansion, expansion), periodic))
            # the full grid, one sector, a cluster's solver
            self.assertFalse(exchange_device.pooled_exchange_applies(orbitals(None), periodic))
            self.assertFalse(exchange_device.pooled_exchange_applies(orbitals(expansion), periodic))
            self.assertFalse(exchange_device.pooled_exchange_applies(orbitals(expansion, expansion), cluster))
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "off"}):
            self.assertFalse(exchange_device.pooled_exchange_applies(orbitals(expansion, expansion), periodic))
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "on"}):
            with self.assertRaises(ValueError):
                exchange_device.exchange_pool_setting()
            # a cluster's solver never reads the switch (refused at setup instead, test below)
            self.assertFalse(exchange_device.pooled_exchange_applies(orbitals(expansion, expansion), cluster))
            with self.assertRaises(ValueError):
                exchange_device.pooled_exchange_applies(orbitals(expansion, expansion), periodic)

    def test_a_wrong_switch_is_refused_at_setup(self) -> None:
        # device_exchange_for (the driver's setup of every device hybrid, clusters too) refuses it before the start
        # SCF, next to the switches of the parity classes; CuPy is not needed for that
        from parsec_python.acceleration.SCF import hybrid as device_hybrid
        from parsec_python.acceleration.backends import cupy as cupy_backend

        folder = Path(tempfile.mkdtemp(prefix="parsec_pool_switch_"))
        self.addCleanup(shutil.rmtree, folder, True)
        problem = _translation(_write_silicon(folder)).problem
        reference = SimpleNamespace(input=problem, grid=SimpleNamespace(volume_element=0.125))
        sectors = SimpleNamespace(device_ids=(0,))
        with mock.patch.object(cupy_backend, "require_cupy", return_value=(None, None)):
            for value in ("on", "1", "tiles"):
                with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": value}):
                    with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_EXCHANGE_POOL must be auto or off"):
                        device_hybrid.device_exchange_for(reference, None, sectors, selected_backend="cupy")
            for value in ("", "auto", "OFF"):
                with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": value}):
                    exchange = device_hybrid.device_exchange_for(reference, None, sectors, selected_backend="cupy")
                    self.assertEqual(exchange.devices, (0,))
            # a semilocal input sets nothing up and reads nothing
            semilocal = replace(problem, scf=replace(problem.scf, xc_functional="pbe", exact_exchange=None))
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "on"}):
                found = device_hybrid.device_exchange_for(
                    SimpleNamespace(input=semilocal), None, sectors, selected_backend="cupy"
                )
                self.assertIsNone(found)

    def test_the_bytes_of_the_pooled_block(self) -> None:
        from parsec_python.acceleration.Hamiltonian import exchange_device

        # N 1000, M 10 (6 occupied); sector A (500 rows, 6 states) on device 0 with its expansion there, sector B
        # (500 rows, 4 states) on device 1 with an expansion of 1234 bytes the home does not hold yet; batch 2
        def expansion(present, size):
            return SimpleNamespace(on_device=lambda device: present, device_bytes=lambda: size)

        sectors = (
            SimpleNamespace(device=0, rows=500, local=np.arange(6), expansion=expansion(True, 99)),
            SimpleNamespace(device=1, rows=500, local=np.arange(4), expansion=expansion(False, 1234)),
        )
        occupations = np.array([1.0] * 6 + [0.0] * 4)
        orbitals = exchange_device.ChannelOrbitals(sectors, occupations, 2.0, 1000, 0.125)
        small = []

        def small_build(channel, *, columns="all", known=None):
            # the pooled block: one full-grid sector of every state
            self.assertEqual(len(channel.sectors), 1)
            self.assertEqual(channel.state_count, 10)
            small.append(columns)
            return bool(small_build.small)

        small_build.small = False
        solvers = SimpleNamespace(batch_size=2, small_build=small_build)
        block, tile, received = 8 * 1000 * 10, 8 * 2 * 9 * 1000, 8 * 2 * 1000
        parts = {0: 8 * 500 * 6, 1: 8 * 500 * 4}
        unfold = fold = 8 * 6 * 2500
        found = exchange_device.pooled_exchange_bytes(orbitals, solvers, (0, 1), columns="occupied")
        self.assertEqual(small, ["occupied"])
        self.assertEqual(found[1], parts[1] + tile)
        self.assertEqual(found[0], block + 1234 + max(unfold, block + tile + received, block + fold + parts[0]))
        self.assertEqual(found[0], block + 1234 + block + tile + received)
        # a small build: the home alone builds; device 1 keeps its sector's W_G only
        small_build.small = True
        found = exchange_device.pooled_exchange_bytes(orbitals, solvers, (0, 1))
        self.assertEqual(found, {0: block + 1234 + block + fold + parts[0], 1: parts[1]})
        # one device given: no small-build test
        small.clear()
        self.assertEqual(set(exchange_device.pooled_exchange_bytes(orbitals, solvers, (0,))), {0, 1})
        self.assertEqual(small, [])
        # the fit: every device against its free bytes and 2 % of its memory
        small_build.small = False
        need = exchange_device.pooled_exchange_bytes(orbitals, solvers, (0, 1))
        total = 10**9
        margin = exchange_device.POOL_MEMORY_MARGIN * total
        for device in (0, 1):
            for spare, expected in ((math.ceil(margin), True), (math.floor(margin) - 1, False)):
                memory = {d: (need[d] + (spare if d == device else 10**8), total) for d in (0, 1)}
                with mock.patch.object(exchange_device, "_device_memory", side_effect=memory.__getitem__):
                    self.assertEqual(exchange_device.pooled_exchange_fits(orbitals, solvers, (0, 1)), expected)

    def test_the_driver_takes_a_hybrid_into_the_sectors_in_one_process(self) -> None:
        from parsec_python.acceleration import driver

        problem = SimpleNamespace(scf=SimpleNamespace(is_hybrid=True))
        with mock.patch.object(route_module, "sector_basis_reached", return_value=True):
            for setting, expected in (("auto", True), ("off", False), ("kpoint", False)):
                with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": setting}):
                    self.assertEqual(driver._gamma_hybrid_in_sectors(problem, None), expected, setting)
                    self.assertFalse(driver._gamma_hybrid_in_sectors(problem, object()), setting)
            not_hybrid = SimpleNamespace(scf=SimpleNamespace(is_hybrid=False))
            self.assertFalse(driver._gamma_hybrid_in_sectors(not_hybrid, None))
        with mock.patch.object(route_module, "sector_basis_reached", return_value=False):
            self.assertFalse(driver._gamma_hybrid_in_sectors(problem, None))

    def test_a_small_cell_keeps_the_full_grid(self) -> None:
        # the 8-atom cube at 12^3 points and 22 states: 38,016 entries, below 2^24
        folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_floor_"))
        self.addCleanup(shutil.rmtree, folder, True)
        translation = _translation(_write_silicon(folder))
        self.assertFalse(route_module.sector_basis_reached(translation.problem))
        with mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 12**3 * 22):
            self.assertTrue(route_module.sector_basis_reached(translation.problem))
        with mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 12**3 * 22 + 1):
            self.assertFalse(route_module.sector_basis_reached(translation.problem))


class _FullGridSolvers:
    """The Kronecker solvers of a cluster with the interface of a periodic cell's (``small_build``, no parity
    classes): sector orbitals then take the pooled block (``PARSEC_CUPY_EXCHANGE_POOL=auto``) or the tiles of the
    full grid, and the two agree to round-off."""

    def __init__(self, solvers) -> None:
        self._solvers = solvers

    def __getattr__(self, name):
        return getattr(self._solvers, name)

    @staticmethod
    def small_build(orbitals, *, columns="all", known=None) -> bool:
        return False

    @staticmethod
    def parity_layout():
        return None


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class PooledExchangeDeviceTests(unittest.TestCase):
    """The pooled block on a GPU: consumed by the build that completes it, and the sector tiles where it does not
    fit the free memory of its device."""

    def setUp(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import DevicePairSolvers
        from parsec_python.acceleration.tests.test_exchange_parity import sector_channel
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        self.orbitals, _ = sector_channel(grid, "D2")
        self.assertGreater(len(self.orbitals.sectors), 1)
        self.solvers = _FullGridSolvers(DevicePairSolvers(grid, (("coulomb", 1.0, None),), 6, 4))

    def _build(self, setting: str):
        """The occupied build and the build of every column that completes it, under ``setting``."""

        from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange

        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": setting}):
            occupied = apply_device_exchange(self.orbitals, self.solvers, (0,), columns="occupied")
            return occupied, apply_device_exchange(self.orbitals, self.solvers, (0,), columns="all", known=occupied)

    def _host(self, application):
        import cupy as cp

        return [cp.asnumpy(part) for part in application.applied]

    def test_the_build_of_every_column_consumes_the_pooled_block(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange

        occupied, tiles = self._build("off")
        self.assertIsNone(occupied.pooled)
        reference = self._host(tiles)
        self._build("auto")  # the device arrays the builds keep (expansions, plans) are made here
        cp.cuda.Device(0).synchronize()
        pool = cp.get_default_memory_pool()
        before = pool.used_bytes()
        self.solvers.take_paths()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "auto"}):
            occupied = apply_device_exchange(self.orbitals, self.solvers, (0,), columns="occupied")
            inner = occupied.pooled
            self.assertIsNotNone(inner)
            block = inner.orbitals.sectors[0]
            self.assertEqual(block.vectors.shape, (self.orbitals.full_size, self.orbitals.state_count))
            application = apply_device_exchange(self.orbitals, self.solvers, (0,), columns="all", known=occupied)
        cp.cuda.Device(0).synchronize()
        after = pool.used_bytes()
        # both builds counted as pooled (the solver's description), their full-grid tiles as such
        self.assertEqual(self.solvers.take_paths(), Counter({"pool": 2, "full": 2}))
        # consumed: nothing of the occupied build is left, the result keeps no block
        self.assertIsNone(occupied.pooled)
        self.assertTrue(all(part is None for part in occupied.applied))
        self.assertIsNone(block.vectors)
        self.assertTrue(all(part is None for part in inner.applied))
        self.assertIsNone(application.pooled)
        # only the result's W_G stays on the device (CuPy's pool rounds each block up to 512 bytes); the block and
        # its W were 2 x 8 N M bytes
        kept = sum(int(part.nbytes) for part in application.applied)
        self.assertLessEqual(after - before, kept + 512 * len(application.applied))
        self.assertGreater(2 * 8 * self.orbitals.full_size * self.orbitals.state_count, 512 * 4)
        # the same operator as the tiles of every sector
        scale = max(float(np.abs(part).max()) for part in reference)
        for pooled_part, tile_part in zip(self._host(application), reference):
            self.assertLess(float(np.abs(pooled_part - tile_part).max()), 1e-12 * scale)
        self.assertAlmostEqual(application.energy, tiles.energy, delta=1e-12 * abs(tiles.energy))
        self.assertEqual(application.columns, tiles.columns)
        # an application of every column keeps no block either
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "auto"}):
            self.assertIsNone(apply_device_exchange(self.orbitals, self.solvers, (0,), columns="all").pooled)

    def test_the_tiles_where_the_block_does_not_fit(self) -> None:
        from parsec_python.acceleration.Hamiltonian import exchange_device

        _occupied, tiles = self._build("off")
        reference = self._host(tiles)
        need = exchange_device.pooled_exchange_bytes(self.orbitals, self.solvers, (0,), columns="occupied")
        self.assertEqual(set(need), {0})
        total = 80 * 2**30
        margin = exchange_device.POOL_MEMORY_MARGIN * total

        def free(spare):
            return mock.patch.object(exchange_device, "_device_memory", return_value=(need[0] + spare, total))

        self.solvers.take_paths()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "auto"}):
            with free(math.ceil(margin)):
                self.assertTrue(
                    exchange_device.pooled_exchange_fits(self.orbitals, self.solvers, (0,), columns="occupied")
                )
            self.assertNotIn("pool-fallback", self.solvers._paths)
            with free(math.floor(margin) - 1):
                self.assertFalse(
                    exchange_device.pooled_exchange_fits(self.orbitals, self.solvers, (0,), columns="occupied")
                )
                occupied = exchange_device.apply_device_exchange(self.orbitals, self.solvers, (0,), columns="occupied")
            self.assertIsNone(occupied.pooled)
            self.assertIn("pool-fallback", self.solvers._paths)
            # the build that completes it keeps the tiles, though the block would fit now (keyed on known.pooled)
            application = exchange_device.apply_device_exchange(
                self.orbitals, self.solvers, (0,), columns="all", known=occupied
            )
        self.assertIsNone(application.pooled)
        # the build and its completion counted as falling back (the solver's description: "in 2 of 2 builds")
        self.assertEqual(self.solvers.take_paths(), Counter({"pool-fallback": 2, "full": 2}))
        # the tiles of the switch off, bit for bit
        for fallback, tile_part in zip(self._host(application), reference):
            np.testing.assert_array_equal(fallback, tile_part)
        self.assertEqual(application.energy, tiles.energy)
        self.assertEqual(application.pair_solves, tiles.pair_solves)


def _periodic_solvers():
    from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import DevicePeriodicPairSolvers

    kernel = SimpleNamespace(shape=(4, 4, 4), values=lambda q, xp=np: xp.zeros((4, 4, 4)))
    return DevicePeriodicPairSolvers(kernel, SimpleNamespace(shape=(4, 4, 4), flat=None), 8)


def _fallback_note(fallback: int, builds: int) -> str:
    return (
        f" (of the symmetry sectors in {fallback} of {builds} builds, where the full-grid block of the orbitals did "
        "not fit the device)"
    )


class FallbackDescriptionTests(unittest.TestCase):
    def test_the_periodic_pair_solver_names_the_fallback(self) -> None:
        solvers = _periodic_solvers()
        plain = solvers.description
        self.assertTrue(plain.endswith("devices take tiles of 8-column orbital blocks"))
        solvers.record_path("full")
        solvers.record_path("pool")
        self.assertEqual(solvers.description, plain)
        solvers.record_path("pool-fallback")
        self.assertEqual(solvers.description, plain + _fallback_note(1, 2))
        # the record is taken (the Fock loop names the builds of each step) and then holds nothing
        taken = solvers.take_paths()
        self.assertEqual(taken, Counter({"full": 1, "pool": 1, "pool-fallback": 1}))
        self.assertEqual(solvers.description, plain)
        self.assertEqual(solvers.describe(taken), plain + _fallback_note(1, 2))
        self.assertEqual(solvers.describe(Counter({"pool-fallback": 3})), plain + _fallback_note(3, 3))
        self.assertEqual(solvers.describe(Counter({"pool": 3, "full": 3})), plain)


class SolverNameTests(unittest.TestCase):
    """The pair solver the records of the Fock loop name (``run_hybrid_scf``): each step the builds of that step, the
    summary those of the run, and a later run with the cached solver (the next structure of a relaxation) nothing
    of an earlier one.  The H4 cluster on the host, its builds recorded as a device build of sector orbitals would
    record them in the periodic pair solver (``DeviceFockEngine``'s naming methods)."""

    def _engine(self, solvers):
        from parsec_python.acceleration.SCF.hybrid import DeviceFockEngine

        class Engine:
            """The host exchange whose builds are pooled while ``fits(build number)`` says so (else the tiles)."""

            description = DeviceFockEngine.description
            take_exchange_paths = DeviceFockEngine.take_exchange_paths
            describe_exchange = DeviceFockEngine.describe_exchange

            def __init__(self, host) -> None:
                self.host = host
                self.solvers = solvers
                self.fits = lambda build: True
                self.builds = 0

            def __getattr__(self, name):
                return getattr(self.host, name)

            def exchange(self, sets, *, columns="occupied", known=None):
                self.builds += 1
                solvers.record_path("pool" if self.fits(self.builds) else "pool-fallback")
                solvers.record_path("full")  # the tiles of the full-grid block, or of the sectors
                return self.host.exchange(sets, columns=columns, known=known)

        return Engine

    def test_the_records_name_the_builds_of_their_step_and_run(self) -> None:
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF import prepare_single_point
        from parsec_python.SCF.hybrid import HostExchange, hybrid_definition, run_hybrid_scf
        from parsec_python.models import ExchangeLoopIteration
        from parsec_python.tests.unit.test_fock_controls import FockLoopControlTests

        solvers = _periodic_solvers()
        plain = solvers.description
        directory = Path(tempfile.mkdtemp(prefix="parsec_solver_names_"))
        self.addCleanup(shutil.rmtree, directory, True)
        library = Path(__file__).resolve().parents[3] / "pp_generation" / "NCPPs" / "PBE" / "H" / "PARSEC"
        shutil.copy(library / "H_POTRE.DAT", directory / "H_POTRE.DAT")
        (directory / "parsec.in").write_text(FockLoopControlTests.BASE.format(extra=""))
        system = prepare_single_point(parse_parsec_input(directory / "parsec.in").problem)
        host = HostExchange(system, hybrid_definition(system.input.scf))
        engine = self._engine(solvers)(host)

        def run():
            records = []
            engine.builds = 0
            result = run_hybrid_scf(system, engine=engine, callback=records.append)
            self.assertTrue(result.converged)
            return result.exact_exchange, [record for record in records if isinstance(record, ExchangeLoopIteration)]

        # the free memory drops after the first build: every later build takes the sector tiles
        engine.fits = lambda build: build == 1
        summary, records = run()
        builds = engine.builds
        self.assertGreater(len(records), 2)
        self.assertEqual(records[0].pair_solver, plain)
        for record in records[1:]:
            self.assertRegex(record.pair_solver, r" \(of the symmetry sectors in (\d+) of \1 builds, where ")
        self.assertEqual(summary.pair_solver, plain + _fallback_note(builds - 1, builds))
        self.assertEqual(sum(int(re.search(r" in (\d+) of", record.pair_solver).group(1)) for record in records[1:]),
                         builds - 1)
        # a fallback recorded after it (another structure) and a run whose builds all pool: no note anywhere
        solvers.record_path("pool-fallback")
        engine.fits = lambda build: True
        summary, records = run()
        self.assertEqual([record.pair_solver for record in records], [plain] * len(records))
        self.assertEqual(summary.pair_solver, plain)
        # the first build falls back only: that record and the summary
        engine.fits = lambda build: build != 1
        summary, records = run()
        self.assertEqual(records[0].pair_solver, plain + _fallback_note(1, 1))
        self.assertEqual([record.pair_solver for record in records[1:]], [plain] * (len(records) - 1))
        self.assertEqual(summary.pair_solver, plain + _fallback_note(1, engine.builds))

    def test_an_engine_without_a_record_names_its_description(self) -> None:
        from parsec_python.SCF.hybrid import _SolverNames

        engine = SimpleNamespace(description="host solver")
        names = _SolverNames(engine)
        self.assertFalse(names.tracked)
        self.assertEqual(names.step(), "host solver")
        engine.description = "changed"
        # read once, after the first build, as before
        self.assertEqual(names.step(), "host solver")
        self.assertEqual(names.run(), "host solver")
        # a device engine whose pair solvers keep no record (a test double, the MPI runner's doubles)
        untracked = SimpleNamespace(
            description="device solver", take_exchange_paths=lambda: None, describe_exchange=lambda paths: "x"
        )
        self.assertFalse(_SolverNames(untracked).tracked)


if __name__ == "__main__":
    unittest.main()
