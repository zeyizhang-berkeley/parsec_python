"""TS, LDA+U, meta-GGA and libxc of bulk cells at the Gamma point in the symmetry sectors (``SCF/gamma_features.py``).

The switches ``PARSEC_GAMMA_TS_SECTORS``, ``PARSEC_GAMMA_LDAU_SECTORS``, ``PARSEC_GAMMA_META_GGA_SECTORS`` and
``PARSEC_GAMMA_LIBXC_SECTORS`` let each feature take the route of the Gamma sectors
(``driver.periodic_sector_refusal(..., features=True)``: the driver in one process and the spin route of the command
line); ``off`` refuses it as before, and so do the callers without ``features`` (the MPI runner, the supercell of a
k-point mesh).  A TS input says ``Ignore_Symmetry: true`` (PARSEC error 159); the command line reads it as that
requirement under ``auto`` or ``on`` (the default of TS is ``off``) where the sectors of a CuPy run take the cell.
On a GPU the sectors give the energies and forces of the full grid.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.acceleration.SCF import gamma_features
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available

_ROOT = Path(__file__).resolve().parents[4]
_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_SI_PP = _ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT"

_SWITCHES = (
    gamma_features.TS_SWITCH,
    gamma_features.LDAU_SWITCH,
    gamma_features.META_GGA_SWITCH,
    gamma_features.LIBXC_SWITCH,
)

_CELL = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side:.6f} {side:.6f} {side:.6f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing:.10f}
Expansion_Order: 8
States_Num: {states}
Fermi_Temp: {temperature}
Max_Iter: 150
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-10
Eigensolver: chebff
Output_Level: 2
{functional}
{extra}Atom_Types_Num: 1
Atom_Type: {symbol}
Local_Component: {local}
{atom_extra}begin Atom_Coord
{coordinates}end Atom_Coord
"""

_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])


def _coordinates(positions) -> str:
    return "".join(f"  {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in positions)


def write_silicon(
    folder: Path, functional: str = "Correlation_Type: pbe", extra: str = "", *, moved: float = 0.0
) -> Path:
    """The 8-atom Si cube (24 points a side), ``moved`` bohr of the atom at the origin along z."""

    positions = np.vstack([_FCC, _FCC + 0.25]) * 10.26
    positions[0, 2] += moved
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_SI_PP, folder / "Si_POTRE.DAT")
    path = folder / "parsec.in"
    path.write_text(
        _CELL.format(
            side=10.26, spacing=10.26 / 24, states=24, temperature=300, functional=functional, extra=extra,
            symbol="Si", local="p", atom_extra="", coordinates=_coordinates(positions),
        )
    )
    return path


def write_silver(folder: Path, extra: str = "", *, moved: float = 0.0) -> Path:
    """The 4-atom fcc Ag cube (16 points a side) with LDA+U (U = 4 eV), ``moved`` bohr of one atom along z."""

    positions = _FCC * 7.729
    positions[0, 2] += moved
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_DATA / "Ag_POTRE.DAT", folder / "Ag_POTRE.DAT")
    path = folder / "parsec.in"
    path.write_text(
        _CELL.format(
            side=7.729, spacing=7.729 / 16, states=34, temperature=1000, functional="Correlation_Type: ca",
            extra=extra, symbol="Ag", local="s", atom_extra="ldaplusu_u 4.0 eV\n", coordinates=_coordinates(positions),
        )
    )
    return path


def write_oxygen(folder: Path, functional: str = "Correlation_Type: pbe", extra: str = "") -> Path:
    """O2 along z about the origin of a spin-polarized cube of 9.9 bohr (``test_gamma_spin_sectors``)."""

    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_DATA / "spin_ca_O_POTRE.DAT", folder / "O_POTRE.DAT")
    path = folder / "parsec.in"
    path.write_text(
        _CELL.format(
            side=9.9, spacing=0.55, states=10, temperature=80, functional=functional,
            extra="Spin_Polarization: .true.\n" + extra, symbol="O", local="p", atom_extra="",
            coordinates=_coordinates([(0.0, 0.0, 1.1414), (0.0, 0.0, -1.1414)]),
        )
    )
    return path


TS = "Van_Der_Waals: .true.\nIgnore_Symmetry: .true.\n"


def _translation(path: Path):
    from parsec_python.Input import parse_parsec_input

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


def _all(value: str) -> dict:
    return {name: value for name in _SWITCHES}


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for name in _SWITCHES:
            for value, expected in (
                ("auto", "auto"), ("ON", "on"), (" OFF ", "off"), ("0", "off"), ("false", "off"), ("no", "off")
            ):
                with mock.patch.dict(os.environ, {name: value}):
                    self.assertEqual(gamma_features.gamma_feature_setting(name), expected)
            with mock.patch.dict(os.environ, {name: ""}):
                self.assertEqual(gamma_features.gamma_feature_setting(name), gamma_features.DEFAULTS[name])
            for wrong in ("1", "true", "sectors"):
                with mock.patch.dict(os.environ, {name: wrong}):
                    with self.assertRaisesRegex(ValueError, f"{name} must be auto, on or off"):
                        gamma_features.gamma_feature_settings()

    def test_floors(self) -> None:
        # measured on A100 (gamma_features.MINIMUM_ENTRIES): TS and meta-GGAs from 2^27, libxc from 2^28, LDA+U (one
        # atom per orbit, weighted sectors) from 2^26 since 2026-11-21 on any number of devices
        self.assertEqual(set(gamma_features.MINIMUM_ENTRIES), set(_SWITCHES))
        self.assertEqual(gamma_features.MINIMUM_ENTRIES[gamma_features.TS_SWITCH], 1 << 27)
        self.assertEqual(gamma_features.MINIMUM_ENTRIES[gamma_features.META_GGA_SWITCH], 1 << 27)
        self.assertEqual(gamma_features.MINIMUM_ENTRIES[gamma_features.LDAU_SWITCH], 1 << 26)
        self.assertEqual(gamma_features.MINIMUM_ENTRIES[gamma_features.LIBXC_SWITCH], 1 << 28)
        self.assertEqual(gamma_features.MINIMUM_DEVICES, {})

    def test_defaults(self) -> None:
        # a meta-GGA, libxc, LDA+U and TS auto (since 2026-11-21: LDA+U with one atom per orbit in the weighted
        # sectors; the Ignore_Symmetry of a TS input read as PARSEC's requirement)
        self.assertEqual(
            gamma_features.DEFAULTS,
            {
                gamma_features.TS_SWITCH: "auto",
                gamma_features.LDAU_SWITCH: "auto",
                gamma_features.META_GGA_SWITCH: "auto",
                gamma_features.LIBXC_SWITCH: "auto",
            },
        )
        with mock.patch.dict(os.environ):
            for name in _SWITCHES:
                os.environ.pop(name, None)
            self.assertEqual(gamma_features.gamma_feature_settings(), gamma_features.DEFAULTS)

    def test_no_gpu_is_probed_on_the_cpu(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}), mock.patch(
            "parsec_python.acceleration.backends.cupy.cupy_available", side_effect=AssertionError("probed")
        ):
            self.assertEqual(gamma_features.process_device_count(), 0)


class RefusalTests(unittest.TestCase):
    """``periodic_sector_refusal`` with and without ``features``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_features_"))
        cls.addClassCleanup(shutil.rmtree, cls.folder, True)
        cls.silicon = _translation(write_silicon(cls.folder / "si")).problem
        cls.silver = _translation(write_silver(cls.folder / "ag")).problem
        cls.oxygen = _translation(write_oxygen(cls.folder / "o2")).problem

    def refusal(self, problem, environment, **keywords):
        from parsec_python.acceleration.driver import periodic_sector_refusal

        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", **environment}):
            return periodic_sector_refusal(problem, **keywords)

    def test_each_feature_follows_its_switch(self) -> None:
        scf = self.silicon.scf
        cases = {
            gamma_features.TS_SWITCH: (replace(self.silicon, scf=replace(scf, van_der_waals=True)), "Tkatchenko"),
            gamma_features.LDAU_SWITCH: (self.silver, "LDA+U"),
            gamma_features.META_GGA_SWITCH: (replace(self.silicon, scf=replace(scf, xc_functional="scan")), "meta-GGA"),
            gamma_features.LIBXC_SWITCH: (
                replace(self.silicon, scf=replace(scf, xc_functional="libxc:gga_x_pbe+gga_c_pbe")),
                "libxc",
            ),
        }
        self.assertIsNone(self.refusal(self.silicon, _all("off")))
        for switch, (problem, reason) in cases.items():
            with self.subTest(switch=switch):
                # off, or a caller without features (the MPI runner, the supercell of a mesh): the reason of before
                self.assertIn(reason, self.refusal(problem, _all("off"), features=True))
                self.assertIn(reason, self.refusal(problem, _all("on")))
                self.assertIn(reason, self.refusal(problem, {**_all("on"), switch: "off"}, features=True))
                # its own switch alone
                self.assertIsNone(self.refusal(problem, {**_all("off"), switch: "on"}, features=True))
                # auto: the basis of these cells (Si 13,824 points x 24 states, Ag 4,096 x 34) is below the floor,
                # which a floor at its size lets in
                entries = gamma_features.basis_entries(problem)
                self.assertEqual(entries, 13824 * 24 if problem is not self.silver else 4096 * 34)
                below = self.refusal(problem, {**_all("off"), switch: "auto"}, features=True)
                self.assertIn(reason, below)
                self.assertIn(f"a basis below {gamma_features.MINIMUM_ENTRIES[switch]} grid points x states", below)
                with mock.patch.object(gamma_features, "process_device_count", return_value=4):
                    with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {switch: entries}):
                        self.assertIsNone(self.refusal(problem, {**_all("off"), switch: "auto"}, features=True))
                    with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {switch: entries + 1}):
                        self.assertIsNotNone(self.refusal(problem, {**_all("off"), switch: "auto"}, features=True))
                # any number of devices (LDA+U needed four from 2026-11-19 to 21)
                for devices in (0, 1, 3):
                    counted = mock.patch.object(gamma_features, "process_device_count", return_value=devices)
                    with counted, mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {switch: entries}):
                        self.assertIsNone(self.refusal(problem, {**_all("off"), switch: "auto"}, features=True))

    def test_a_libxc_meta_gga_needs_both_switches(self) -> None:
        problem = replace(
            self.silicon, scf=replace(self.silicon.scf, xc_functional="libxc:mgga_x_r2scan+mgga_c_r2scan")
        )
        self.assertTrue(problem.scf.is_meta_gga)
        both = {gamma_features.META_GGA_SWITCH: "on", gamma_features.LIBXC_SWITCH: "on"}
        self.assertIsNone(self.refusal(problem, both, features=True))
        self.assertIn("meta-GGA", self.refusal(problem, {**both, gamma_features.META_GGA_SWITCH: "off"}, features=True))
        self.assertIn("libxc", self.refusal(problem, {**both, gamma_features.LIBXC_SWITCH: "off"}, features=True))

    def test_hybrids_and_spin_ldau_keep_the_refusal(self) -> None:
        scf = self.silicon.scf
        for changed in (replace(scf, xc_functional="pbe0", van_der_waals=True),):
            problem = replace(self.silicon, scf=changed)
            # the hybrid route of the sectors takes the hybrid, not the TS correction with it
            self.assertIn("Tkatchenko", self.refusal(problem, _all("on"), features=True, hybrid_sectors=True))
        spin_silver = replace(self.silver, scf=replace(self.silver.scf, spin_polarized=True))
        # PARSEC_LDAU_SPIN_SECTORS: auto (the default since 2026-11-21) and on let the spin cell take the spin
        # sectors under the LDA+U switch; off keeps the refusal of before (the 1x1x1 k-point driver)
        for value in ("auto", "on"):
            spin = {gamma_features.LDAU_SPIN_SWITCH: value}
            self.assertIsNone(self.refusal(spin_silver, {**_all("on"), **spin}, features=True))
            self.assertEqual(self.refusal(spin_silver, {**_all("off"), **spin}, features=True), "LDA+U")
        with mock.patch.dict(os.environ):
            os.environ.pop(gamma_features.LDAU_SPIN_SWITCH, None)
            self.assertIsNone(self.refusal(spin_silver, _all("on"), features=True))
        off = {gamma_features.LDAU_SPIN_SWITCH: "off"}
        self.assertEqual(self.refusal(spin_silver, {**_all("on"), **off}, features=True), "LDA+U")
        # the spin cell takes TS, a meta-GGA and libxc under their switches
        oxygen = self.oxygen
        for changed in (
            replace(oxygen.scf, van_der_waals=True),
            replace(oxygen.scf, xc_functional="scan"),
            replace(oxygen.scf, xc_functional="libxc:gga_x_pbe+gga_c_pbe"),
        ):
            problem = replace(oxygen, scf=changed)
            self.assertIsNone(self.refusal(problem, _all("on"), features=True))
            self.assertIsNotNone(self.refusal(problem, _all("off"), features=True))
            self.assertIsNotNone(self.refusal(problem, _all("on")))
            # auto above the floor: a meta-GGA and libxc take the spin sectors; TS keeps the 1x1x1 k-point driver,
            # which was faster
            with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, dict.fromkeys(_SWITCHES, 1)):
                found = self.refusal(problem, _all("auto"), features=True)
            if changed.van_der_waals:
                self.assertEqual(
                    found,
                    "the Tkatchenko-Scheffler dispersion correction (a spin-polarized cell, which keeps the 1x1x1 "
                    "k-point driver, under PARSEC_GAMMA_TS_SECTORS=auto; on takes it)",
                )
            else:
                self.assertIsNone(found)
        # the unpolarized TS cell above the floor under auto takes the sectors
        ts_silicon = replace(self.silicon, scf=replace(scf, van_der_waals=True))
        with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, dict.fromkeys(_SWITCHES, 1)):
            self.assertIsNone(self.refusal(ts_silicon, _all("auto"), features=True))

    def test_the_spin_route_of_the_command_line(self) -> None:
        from parsec_python.acceleration.SCF.gamma_spin import gamma_spin_on_sector_route

        translation = _translation(write_oxygen(self.folder / "o2_ts", extra=TS))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", **_all("on")}):
            self.assertTrue(gamma_spin_on_sector_route(translation, mpi=False, backend_is_cupy=lambda: True))
            self.assertFalse(gamma_spin_on_sector_route(translation, mpi=True, backend_is_cupy=lambda: True))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", **_all("off")}):
            # the 1x1x1 k-point driver, as before
            self.assertFalse(gamma_spin_on_sector_route(translation, mpi=False, backend_is_cupy=lambda: True))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", **_all("auto")}), mock.patch.dict(
            gamma_features.MINIMUM_ENTRIES, dict.fromkeys(_SWITCHES, 1)
        ):
            # TS of a spin cell keeps that driver under auto at every size
            self.assertFalse(gamma_spin_on_sector_route(translation, mpi=False, backend_is_cupy=lambda: True))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto"}):
            for name in _SWITCHES:
                os.environ.pop(name, None)
            # the default (TS auto since 2026-11-21: a spin cell keeps the 1x1x1 driver)
            self.assertEqual(gamma_features.gamma_feature_setting(gamma_features.TS_SWITCH), "auto")
            self.assertFalse(gamma_spin_on_sector_route(translation, mpi=False, backend_is_cupy=lambda: True))

    def test_the_mpi_runner_keeps_refusing(self) -> None:
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _refuse_unsupported_periodic

        translation = _translation(write_silicon(self.folder / "si_scan", "Correlation_Type: scan"))
        with mock.patch.dict(os.environ, _all("auto")):
            with self.assertRaisesRegex(ValueError, "a meta-GGA functional of a periodic cell is not available"):
                _refuse_unsupported_periodic(translation)


class AtomOrbitTests(unittest.TestCase):
    """LDA+U keeps the projectors of every atom: the group of a cell drops the orbits of its atoms."""

    def test_the_orbits_are_dropped(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.driver import _without_atom_orbits

        @dataclass(frozen=True)
        class Reduction:
            group_order: int
            atom_representatives: np.ndarray = None
            atom_orbit_sizes: np.ndarray = None

        with_orbits = Reduction(4, np.array([0, 0, 2, 2]), np.array([2, 2, 2, 2]))
        dropped = _without_atom_orbits(with_orbits)
        self.assertIsNot(dropped, with_orbits)
        self.assertEqual(dropped.group_order, 4)
        self.assertIsNone(dropped.atom_representatives)
        self.assertIsNone(dropped.atom_orbit_sizes)
        # the same object where it knows no orbits (the identity of the group of the origin is kept)
        bare = Reduction(4)
        self.assertIs(_without_atom_orbits(bare), bare)
        other = SimpleNamespace(group_order=2)
        self.assertIs(_without_atom_orbits(other), other)
        with self.assertRaises(RuntimeError):
            _without_atom_orbits(SimpleNamespace(atom_representatives=np.zeros(2), atom_orbit_sizes=np.ones(2)))


class MetaGGAStartTests(unittest.TestCase):
    """The PBE start of a meta-GGA loop of a periodic cell on the orbits of the sectors (``SCF.pbc``'s hook)."""

    def system(self, reference, adapter):
        from parsec_python.acceleration.SCF.single_point import AcceleratedPreparedSinglePointSystem

        return AcceleratedPreparedSinglePointSystem(
            reference=reference, backend=None, backend_info=None, scalar_field_adapter=adapter
        )

    def test_the_hook_on_orbits_and_on_the_full_grid(self) -> None:
        from types import SimpleNamespace

        from parsec_python.V_xc import XCResult

        calls = []

        def hook(density):
            calls.append(np.asarray(density))
            return XCResult(
                potential=2.0 * np.asarray(density), energy_per_electron=np.zeros(4), energy_density=np.zeros(4),
                total_energy=1.0,
            )

        @dataclass(frozen=True)
        class SpinResult:
            potential_up: np.ndarray
            potential_down: np.ndarray

        def spin_hook(up, down):
            return SpinResult(potential_up=np.asarray(up) + 1.0, potential_down=np.asarray(down) - 1.0)

        # an orbit field of two orbits of sizes 3 and 1 on a grid of four points
        adapter = SimpleNamespace(
            to_full=lambda values: np.asarray(values)[[0, 0, 0, 1]],
            from_full=lambda values: np.asarray(values)[[0, 3]],
        )
        reference = SimpleNamespace(evaluate_meta_gga_start_xc=hook, evaluate_meta_gga_start_spin_xc=spin_hook)
        # the full grid: the hook itself, as before
        self.assertIs(self.system(reference, None).evaluate_meta_gga_start_xc, hook)
        result = self.system(reference, adapter).evaluate_meta_gga_start_xc(np.array([1.0, 5.0]))
        np.testing.assert_array_equal(calls[-1], [1.0, 1.0, 1.0, 5.0])
        np.testing.assert_array_equal(result.potential, [2.0, 10.0])
        self.assertEqual(result.total_energy, 1.0)
        spin = self.system(reference, adapter).evaluate_meta_gga_start_spin_xc(
            np.array([1.0, 2.0]), np.array([3.0, 4.0])
        )
        np.testing.assert_array_equal(spin.potential_up, [2.0, 3.0])
        np.testing.assert_array_equal(spin.potential_down, [2.0, 3.0])
        # a cluster's reference has no hook: neither has the system (the loop then takes its own PBE)
        bare = self.system(SimpleNamespace(), adapter)
        self.assertIsNone(getattr(bare, "evaluate_meta_gga_start_xc", None))
        self.assertIsNone(getattr(bare, "evaluate_meta_gga_start_spin_xc", None))


class _Prepared(BaseException):
    """Raised by the stand-in of ``prepare_single_point``: the routing is the subject, not the calculation."""


class CommandLineTests(unittest.TestCase):
    """The symmetry the command line prepares a TS cell with (its preparation is not run)."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.count = 0

    def route(self, path: Path, options=(), environment=None, *, cupy=True):
        """The outcome of the command line on ``path``; ``cupy``: whether its backend resolves to CuPy (the runs
        use ``--device cpu``, the stand-in answers for the TS rule)."""

        seen = {}

        def prepare(problem, **keywords):
            seen.update(keywords)
            raise _Prepared

        errors, output = StringIO(), StringIO()
        with mock.patch.dict(os.environ, environment or {}), mock.patch.object(
            cli_module, "prepare_single_point", side_effect=prepare
        ), mock.patch.object(
            cli_module, "_backend_resolves_to_cupy", return_value=cupy
        ), redirect_stdout(output), redirect_stderr(errors):
            try:
                status = cli_module.main([str(path), "--no-archive", "--device", "cpu", *options])
            except _Prepared:
                log = (path.parent / "parsec.out").read_text() if (path.parent / "parsec.out").exists() else ""
                return ("prepared", seen["symmetry"], log)
        return ("exit", status, errors.getvalue())

    def silicon(self, extra: str = TS) -> Path:
        self.count += 1
        return write_silicon(self.directory / f"case{self.count}", extra=extra)

    def test_ignore_symmetry_of_a_ts_cell_is_parsecs_requirement_under_auto(self) -> None:
        outcome = self.route(self.silicon(), environment={gamma_features.TS_SWITCH: "on"})
        self.assertEqual(outcome[:2], ("prepared", "auto"))
        self.assertIn("NOTE: Ignore_Symmetry of this TS input", outcome[2])
        self.assertIn("PARSEC_GAMMA_TS_SECTORS=on", outcome[2])
        # auto: this small cell is below the floor of the basis and keeps the full grid, a larger one takes them
        outcome = self.route(self.silicon(), environment={gamma_features.TS_SWITCH: "auto"})
        self.assertEqual(outcome[:2], ("prepared", "off"))
        with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {gamma_features.TS_SWITCH: 1}):
            outcome = self.route(self.silicon(), environment={gamma_features.TS_SWITCH: "auto"})
        self.assertEqual(outcome[:2], ("prepared", "auto"))
        # off, --symmetry off, a cell without TS: Ignore_Symmetry keeps the full grid, as before
        for options, environment in (
            ((), {gamma_features.TS_SWITCH: "off"}),
            (("--symmetry", "off"), {gamma_features.TS_SWITCH: "on"}),
        ):
            with self.subTest(options=options, environment=environment):
                outcome = self.route(self.silicon(), options, environment)
                self.assertEqual(outcome[:2], ("prepared", "off"))
                self.assertNotIn("NOTE: Ignore_Symmetry of this TS input", outcome[2])
        outcome = self.route(self.silicon("Ignore_Symmetry: .true.\n"), environment={gamma_features.TS_SWITCH: "on"})
        self.assertEqual(outcome[:2], ("prepared", "off"))
        # forces of the Fortran-derived routines (the routines of every TS input) keep the full grid
        outcome = self.route(self.silicon(TS + "Skip_force: .false.\n"), environment={gamma_features.TS_SWITCH: "on"})
        self.assertEqual(outcome[:2], ("prepared", "off"))

    def test_ignore_symmetry_keeps_its_meaning_where_the_sectors_cannot_take_the_cell(self) -> None:
        # PARSEC_PERIODIC_SYMMETRY=off, a backend that is not CuPy, the switch off (the former default) even for a
        # cell above the floor: no NOTE, the full grid, as in PARSEC 1.5
        cases = (
            ({gamma_features.TS_SWITCH: "on", "PARSEC_PERIODIC_SYMMETRY": "off"}, True),
            ({gamma_features.TS_SWITCH: "on"}, False),
            ({gamma_features.TS_SWITCH: "auto"}, False),
            ({}, False),
            ({gamma_features.TS_SWITCH: "off"}, True),
        )
        for environment, cupy in cases:
            with self.subTest(environment=environment, cupy=cupy), mock.patch.dict(os.environ):
                os.environ.pop(gamma_features.TS_SWITCH, None)
                with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {gamma_features.TS_SWITCH: 1}):
                    outcome = self.route(self.silicon(), environment=environment, cupy=cupy)
                self.assertEqual(outcome[:2], ("prepared", "off"))
                self.assertNotIn("NOTE: Ignore_Symmetry of this TS input", outcome[2])

    def test_the_default_takes_the_sectors_of_an_unpolarized_cell_above_the_floor(self) -> None:
        # since 2026-11-21 the default is auto: Ignore_Symmetry of a TS input is PARSEC's requirement, the cell takes
        # the sectors from the floor on (patched to 1 here), with the NOTE; below it the full grid, as before
        with mock.patch.dict(os.environ):
            os.environ.pop(gamma_features.TS_SWITCH, None)
            with mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {gamma_features.TS_SWITCH: 1}):
                outcome = self.route(self.silicon())
            self.assertEqual(outcome[:2], ("prepared", "auto"))
            self.assertIn("NOTE: Ignore_Symmetry of this TS input", outcome[2])
            self.assertIn("PARSEC_GAMMA_TS_SECTORS=auto", outcome[2])
            outcome = self.route(self.silicon())
            self.assertEqual(outcome[:2], ("prepared", "off"))
            self.assertNotIn("NOTE: Ignore_Symmetry of this TS input", outcome[2])
        # (a spin cell keeps the 1x1x1 k-point driver under auto: RefusalTests and the spin route test above)

    def test_a_wrong_value_is_refused_before_the_scf(self) -> None:
        # (the driver reads every switch at its preparation; the command line through the TS rule)
        outcome = self.route(self.silicon(), environment={gamma_features.TS_SWITCH: "sometimes"})
        self.assertEqual(outcome[0], "exit")
        self.assertIn("PARSEC_GAMMA_TS_SECTORS must be auto, on or off", outcome[2])


def _energy(path: Path) -> float:
    found = re.findall(r"Total Energy[^=\n]*=\s*(-?\d+\.\d+)", path.read_text())
    return float(found[-1])


def _forces(path: Path) -> np.ndarray:
    """The last table of total forces (after the net force is removed) of ``parsec.out``."""

    text = path.read_text()
    marker = "coordinates and tot. forces (after setting net force to zero)"
    rows = []
    for line in text[text.rindex(marker):].splitlines()[1:]:
        if re.match(r"\s*\d+\s+-?\d", line):
            rows.append([float(value) for value in line.split()[4:7]])
        elif rows and line.strip():
            break
    return np.array(rows)


def _device_libxc() -> bool:
    try:
        from parsec_python.acceleration.V_xc.device_libxc import device_libxc_available

        return device_libxc_available()
    except ImportError:  # pragma: no cover
        return False


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class DeviceTests(unittest.TestCase):
    """``main.py`` on a GPU: the full grid (switch off) against the sectors (on), the same energies and forces."""

    def run_both(self, writer, switches, tolerance=2.0e-7):
        folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_features_gpu_"))
        self.addCleanup(shutil.rmtree, folder, True)
        logs = {}
        for setting in ("off", "on"):
            path = writer(folder / setting)
            with mock.patch.dict(os.environ, {name: setting for name in switches}), redirect_stdout(
                StringIO()
            ), redirect_stderr(StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(cli_module.main([str(path), "--no-archive"]), 0, setting)
            logs[setting] = path.parent / "parsec.out"
        self.assertAlmostEqual(_energy(logs["on"]), _energy(logs["off"]), delta=tolerance)
        return {setting: log.read_text() for setting, log in logs.items()}, logs

    def test_ts_single_point(self) -> None:
        texts, _ = self.run_both(lambda folder: write_silicon(folder, extra=TS, moved=0.12), [gamma_features.TS_SWITCH])
        self.assertIn("the full grid is used", texts["off"])
        self.assertIn("NOTE: Ignore_Symmetry of this TS input", texts["on"])
        self.assertRegex(texts["on"], r"orbital_symmetry_representations = [2-9]")

    def test_ldau_single_point(self) -> None:
        # the perfect cell: the commuting involutions with the face-centring translations under
        # PARSEC_LDAU_WEIGHTED_SECTORS=off, whose group knows the orbits of its atoms (one atom per orbit since
        # 2026-11-21, PARSEC_LDAU_PROJECTOR_ORBITS)
        with mock.patch.dict(os.environ, {"PARSEC_LDAU_WEIGHTED_SECTORS": "off"}):
            texts, _ = self.run_both(write_silver, [gamma_features.LDAU_SWITCH])
        self.assertIn("does not take LDA+U", texts["off"])
        self.assertIn("exact commuting group of 32 operations", texts["on"])
        self.assertIn("with PARSEC_GAMMA_LDAU_SECTORS=on", texts["on"])

    def test_ldau_auto_takes_one_device(self) -> None:
        # since 2026-11-21 auto needs no count of devices (four from 2026-11-19 to 21)
        from parsec_python.acceleration.driver import periodic_sector_refusal

        folder = Path(tempfile.mkdtemp(prefix="parsec_gamma_features_gpu_"))
        self.addCleanup(shutil.rmtree, folder, True)
        path = write_silver(folder)
        problem = _translation(path).problem
        floor = mock.patch.dict(gamma_features.MINIMUM_ENTRIES, {gamma_features.LDAU_SWITCH: 1})
        output = StringIO()
        with floor, mock.patch.dict(
            os.environ, {gamma_features.LDAU_SWITCH: "auto", "PARSEC_CUPY_DEVICES": "current"}
        ), redirect_stdout(output), redirect_stderr(StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(gamma_features.process_device_count(), 1)
            self.assertIsNone(periodic_sector_refusal(problem, features=True))
            self.assertEqual(cli_module.main([str(path), "--no-archive"]), 0)
        self.assertIn("with PARSEC_GAMMA_LDAU_SECTORS=auto", (folder / "parsec.out").read_text())

    def test_meta_gga_forces(self) -> None:
        texts, logs = self.run_both(
            lambda folder: write_silicon(folder, "Correlation_Type: scan", "Skip_force: .false.\n", moved=0.12),
            [gamma_features.META_GGA_SWITCH],
            tolerance=5.0e-7,
        )
        self.assertIn("does not take a meta-GGA functional", texts["off"])
        self.assertIn("Forces of the periodic cell in the sectors of its point group", texts["on"])
        np.testing.assert_allclose(_forces(logs["on"]), _forces(logs["off"]), atol=2.0e-5)

    @unittest.skipUnless(_device_libxc(), "a CUDA build of libxc >= 7.1 (PARSEC_LIBXC_LIBRARY) is required")
    def test_libxc_forces(self) -> None:
        texts, logs = self.run_both(
            lambda folder: write_silicon(
                folder, "LibXC_Functional: gga_x_pbe gga_c_pbe", "Skip_force: .false.\n", moved=0.12
            ),
            [gamma_features.LIBXC_SWITCH],
        )
        self.assertIn("does not take a libxc functional", texts["off"])
        np.testing.assert_allclose(_forces(logs["on"]), _forces(logs["off"]), atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
