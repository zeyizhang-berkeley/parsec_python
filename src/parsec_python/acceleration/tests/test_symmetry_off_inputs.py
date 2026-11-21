"""Inputs that switch the point group off in PARSEC 1.5, kept in the sectors of the group they leave intact.

``acceleration/SCF/symmetry_off_inputs.py``: the reasons the parser records, the switches and what they keep,
the plan of the polarizability's field loops and the constrained detection (CPU), the ordered ``polar.dat`` of a
field loop that solves out of order, and on a GPU complete runs of the accelerated CLI under ``auto`` against
``off`` (the full grid of before): ``Polarizability``, ``Add_Point_Charges``, ``Foldblock_Side_Length`` 2, a spin
``Restart_Run`` whose densities keep or break the group, and ``Restart_From_Wfndat``.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.SCF import symmetry_off_inputs as module
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, GridSettings, PointCharge

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
GPU = cupy_available() and native_available()
SWITCHES = (
    module.POLARIZABILITY_SWITCH,
    module.POINT_CHARGE_SWITCH,
    module.FOLDBLOCK_SWITCH,
    module.SPIN_RESTART_SWITCH,
    module.WFNDAT_RESTART_SWITCH,
)


def _h2_input(directory: Path, extra: str = "", *, criterion: str = "1.0e-8") -> Path:
    """H2 along z on a coarse grid (the signed permutations of D4h), with ``extra`` lines."""

    text = f"""\
Boundary_Sphere_Radius: 4.0 bohr
Grid_Spacing: 0.8 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 4
States_Num: 3
Max_Iter: 60
Convergence_Criterion: {criterion}
Eigensolver: chebff
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0  0.0  0.7
  0.0  0.0 -0.7
end Atom_Coord
{extra}
"""
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "water_H_POTRE.DAT").read_bytes())
    return path


_CH4 = """\
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 6
States_Num: 8
Max_Iter: 80
Convergence_Criterion: 1.0e-8
Diag_Tolerance: 1.0e-8
Eigensolver: chebff
Correlation_Type: ca
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 2
Atom_Type: C
Local_Component: p
begin Atom_Coord
  0.0  0.0  0.0
end Atom_Coord
Atom_Type: H
Local_Component: s
begin Atom_Coord
  1.19  1.19  1.19
 -1.19 -1.19  1.19
 -1.19  1.19 -1.19
  1.19 -1.19 -1.19
end Atom_Coord
{extra}
"""


def _ch4_input(directory: Path, extra: str = "") -> Path:
    path = directory / "parsec.in"
    path.write_text(_CH4.format(extra=extra), encoding="utf-8")
    shutil.copyfile(DATA / "spin_ca_C_POTRE.DAT", directory / "C_POTRE.DAT")
    shutil.copyfile(DATA / "spin_ca_H_POTRE.DAT", directory / "H_POTRE.DAT")
    return path


class ParserReasonTests(unittest.TestCase):
    """What the parser records where PARSEC 1.5 switches the point group off."""

    def reasons(self, extra: str) -> tuple[str, ...]:
        with tempfile.TemporaryDirectory() as folder:
            translation = parse_parsec_input(_h2_input(Path(folder), extra))
        return translation.symmetry_off_reasons, translation

    def test_reasons(self) -> None:
        cases = {
            "": (),
            "Polarizability: .true.": ("polarizability",),
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\nbegin Point_Coord\n 0.4 2.0 0.3\nend Point_Coord":
                ("point_charges",),
            "Foldblock_Side_Length: 2": ("foldblock",),
            "Foldblock_Side_Length: 2\nIgnore_Symmetry: .true.": (),
            "Polarizability: .true.\nFoldblock_Side_Length: 2": ("foldblock", "polarizability"),
            "Molecular_Dynamics: .true.": ("molecular_dynamics",),
            "Minimization: manual\nMovement_Num: 0": ("manual",),
        }
        for extra, expected in cases.items():
            with self.subTest(extra=extra):
                reasons, _ = self.reasons(extra)
                self.assertEqual(tuple(sorted(reasons)), tuple(sorted(expected)))

    def test_kept_group_reasons_follow_the_switches(self) -> None:
        _, polar = self.reasons("Polarizability: .true.")
        _, fold = self.reasons("Foldblock_Side_Length: 2")
        _, ignored = self.reasons("Polarizability: .true.\nIgnore_Symmetry: .true.")
        _, dynamics = self.reasons(
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\nbegin Point_Coord\n 0.0 0.0 2.4\n"
            "end Point_Coord\nMolecular_Dynamics: .true."
        )
        with patch.dict(os.environ, {name: "auto" for name in SWITCHES}):
            # the coarse H2 (about 520 points times 3 states) is below the work of the sectors of auto
            self.assertEqual(module.kept_group_reasons(polar), ())
            with patch.object(module, "POLARIZABILITY_SECTOR_WORK", 1.0e3):
                self.assertEqual(module.kept_group_reasons(polar), ("polarizability",))
            self.assertEqual(module.kept_group_reasons(fold), ("foldblock",))
            self.assertEqual(module.kept_group_reasons(ignored), ())
            # one reason without a group of its own keeps the full grid
            self.assertEqual(module.kept_group_reasons(dynamics), ())
        with patch.dict(os.environ, {module.POLARIZABILITY_SWITCH: "on"}):
            self.assertEqual(module.kept_group_reasons(polar), ("polarizability",))
            self.assertEqual(module.kept_group_reasons(ignored), ())
            self.assertIn("PARSEC_POLARIZABILITY_SYMMETRY=on", module.kept_group_note(("polarizability",)))
        with patch.dict(os.environ, {module.POLARIZABILITY_SWITCH: "off"}):
            self.assertEqual(module.kept_group_reasons(polar), ())
        with patch.dict(os.environ, {module.FOLDBLOCK_SWITCH: "off"}):
            self.assertEqual(module.kept_group_reasons(fold), ())

    def test_work_estimate(self) -> None:
        _, polar = self.reasons("Polarizability: .true.")
        grid = build_cluster_grid(polar.problem.grid)
        work = module.estimated_full_grid_work(polar.problem)
        # the volume of the sphere over h^3 against the points of the grid, times States_Num 3
        self.assertAlmostEqual(work / (3 * grid.size), 1.0, delta=0.25)
        # without States_Num: half the electrons (two of H2) plus six, as the SCF plans them (the pseudopotentials
        # are read for the electrons; without their files no estimate)
        from dataclasses import replace

        self.assertIsNone(
            module.estimated_full_grid_work(replace(polar.problem, scf=replace(polar.problem.scf, number_of_states=None)))
        )
        with tempfile.TemporaryDirectory() as folder:
            problem = parse_parsec_input(_h2_input(Path(folder), "Polarizability: .true.")).problem
            default = replace(problem, scf=replace(problem.scf, number_of_states=None))
            self.assertAlmostEqual(module.estimated_full_grid_work(default) / work, 7.0 / 3.0, places=12)

    def test_unknown_values_are_refused(self) -> None:
        for name in SWITCHES:
            with self.subTest(name=name), patch.dict(os.environ, {name: "maybe"}):
                with self.assertRaises(ValueError) as caught:
                    module.read_settings()
                self.assertIn(name, str(caught.exception))
        with patch.dict(os.environ, {name: "" for name in SWITCHES}):
            # every default auto but that of wfn.dat restarts (the sectors take none of its orbitals)
            settings = module.read_settings()
            self.assertEqual(settings.pop("wfndat_restart"), "off")
            self.assertEqual(set(settings.values()), {"auto"})

    def test_unknown_value_stops_the_cli_before_the_run(self) -> None:
        from parsec_python.acceleration.cli import main

        with tempfile.TemporaryDirectory() as folder:
            path = _h2_input(Path(folder), "Polarizability: .true.")
            stream = StringIO()
            with patch.dict(os.environ, {module.POLARIZABILITY_SWITCH: "maybe"}), redirect_stdout(
                stream
            ), redirect_stderr(stream):
                status = main([str(path), "--quiet", "--dry-run"])
        self.assertEqual(status, 2)
        self.assertIn("PARSEC_POLARIZABILITY_SYMMETRY", stream.getvalue())

    def test_restart_switches(self) -> None:
        def translation(*, spin, wfndat, restart=True, periodic=None):
            return SimpleNamespace(
                restart_run=restart,
                restart_from_wfndat=wfndat,
                problem=SimpleNamespace(periodic_cell=periodic, scf=SimpleNamespace(spin_polarized=spin)),
            )

        large = patch.object(module, "estimated_full_grid_work", lambda problem: 2.0 * module.SPIN_RESTART_SECTOR_WORK)
        small = patch.object(module, "estimated_full_grid_work", lambda problem: 0.5 * module.SPIN_RESTART_SECTOR_WORK)
        with patch.dict(os.environ, {name: "auto" for name in SWITCHES}), large:
            self.assertEqual(module.restart_sector_switch(translation(spin=True, wfndat=False)), module.SPIN_RESTART_SWITCH)
            self.assertIsNone(module.restart_sector_switch(translation(spin=False, wfndat=False)))
            self.assertEqual(
                module.restart_sector_switch(translation(spin=False, wfndat=True)), module.WFNDAT_RESTART_SWITCH
            )
            self.assertIn(module.SPIN_RESTART_SWITCH, module.restart_sector_switch(translation(spin=True, wfndat=True)))
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=False, restart=False)))
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=False, periodic=object())))
        with patch.dict(os.environ, {name: "auto" for name in SWITCHES}), small:
            # a small spin restart under auto: the full grid of off (the sectors are slower there)
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=False)))
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=True)))
            # a closed-shell wfn.dat restart has no size rule
            self.assertEqual(
                module.restart_sector_switch(translation(spin=False, wfndat=True)), module.WFNDAT_RESTART_SWITCH
            )
        with patch.dict(os.environ, {module.SPIN_RESTART_SWITCH: "on", module.WFNDAT_RESTART_SWITCH: "auto"}), small:
            self.assertEqual(module.restart_sector_switch(translation(spin=True, wfndat=False)), module.SPIN_RESTART_SWITCH)
            self.assertIn(module.SPIN_RESTART_SWITCH, module.restart_sector_switch(translation(spin=True, wfndat=True)))
            self.assertIn(
                "PARSEC_SPIN_RESTART_SECTORS=on", module._switch_values(module.restart_sector_switch(translation(spin=True, wfndat=True)))
            )
        with patch.dict(os.environ, {module.SPIN_RESTART_SWITCH: "off"}), large:
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=False)))
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=True)))
        with patch.dict(os.environ, {module.WFNDAT_RESTART_SWITCH: "off"}):
            self.assertIsNone(module.restart_sector_switch(translation(spin=False, wfndat=True)))
        with patch.dict(os.environ, {name: "" for name in SWITCHES}), large:
            # the defaults: a large spin parsec.dat restart in the sectors, wfn.dat on the full grid
            self.assertEqual(module.restart_sector_switch(translation(spin=True, wfndat=False)), module.SPIN_RESTART_SWITCH)
            self.assertIsNone(module.restart_sector_switch(translation(spin=False, wfndat=True)))
            self.assertIsNone(module.restart_sector_switch(translation(spin=True, wfndat=True)))
        with patch.dict(os.environ, {module.SPIN_RESTART_SWITCH: "maybe"}):
            with self.assertRaises(ValueError):
                module.spin_restart_sectors_setting()

    def test_spin_restart_work_estimate(self) -> None:
        """The O2 fixture is far below the size of the sectors of a spin restart under auto."""

        from parsec_python.tests.spin_reference import load_cases, write_case

        with tempfile.TemporaryDirectory() as folder:
            path = write_case(dict(load_cases()["o2_ca"]), Path(folder), extra=("Restart_Run: .true.",))
            translation = parse_parsec_input(path)
            with patch.dict(os.environ, {module.SPIN_RESTART_SWITCH: "auto"}):
                self.assertLess(module.estimated_full_grid_work(translation.problem), module.SPIN_RESTART_SECTOR_WORK)
                self.assertIsNone(module.restart_sector_switch(translation))
            with patch.dict(os.environ, {module.SPIN_RESTART_SWITCH: "on"}):
                self.assertEqual(module.restart_sector_switch(translation), module.SPIN_RESTART_SWITCH)


_CH4_RELAX = """\
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 6
States_Num: 8
Max_Iter: 80
Convergence_Criterion: 1.0e-6
Eigensolver: chebff
Correlation_Type: ca
Hartree_Boundary_Tolerance: off
Minimization: bfgs
Movement_Num: 2
Force_Min: 0.0001
Atom_Types_Num: 2
Atom_Type: C
Local_Component: p
Move_Flag: -1
begin Atom_Coord
  0.0  0.0  0.0  {c}
end Atom_Coord
Atom_Type: H
Local_Component: s
Move_Flag: -1
begin Atom_Coord
  1.19  1.19  1.19  {h}
 -1.19 -1.19  1.19  1
 -1.19  1.19 -1.19  1
  1.19 -1.19 -1.19  1
end Atom_Coord
{extra}
"""

#: Two charges on the z axis of CH4: the group of the atoms and charges is D2d (of the Td of the atoms).
_CHARGES_Z = (
    "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\nbegin Point_Coord\n 0.0 0.0 2.4\n 0.0 0.0 -2.4\n"
    "end Point_Coord"
)

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_AU_SOC = """\
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
Atom_Types_Num: 1
Total_Atom_Num: 1
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   0 0 0
End atom_coord
Correlation_Type: CA
{extra}
"""


def _ch4_relax_input(directory: Path, extra: str = "", *, fixed: str = "h") -> Path:
    """CH4 relaxed by BFGS; ``fixed`` names the atom Move_Flag fixes: ``h`` (one H, which every operation of Td
    but those of its threefold axis exchanges with a movable H), ``c`` (the C, an orbit of its own) or ``none``."""

    path = directory / "parsec.in"
    path.write_text(
        _CH4_RELAX.format(extra=extra, c=0 if fixed == "c" else 1, h=0 if fixed == "h" else 1), encoding="utf-8"
    )
    shutil.copyfile(DATA / "spin_ca_C_POTRE.DAT", directory / "C_POTRE.DAT")
    shutil.copyfile(DATA / "spin_ca_H_POTRE.DAT", directory / "H_POTRE.DAT")
    return path


def _au_soc_input(directory: Path, extra: str = "") -> Path:
    path = directory / "parsec.in"
    path.write_text(_AU_SOC.format(extra=extra), encoding="utf-8")
    shutil.copyfile(_AU_POTRE, directory / "Au_POTRE.DAT")
    return path


class KeptGroupGuardTests(unittest.TestCase):
    """Inputs whose group another rule of the run gives up keep no group (review of 2026-11-19)."""

    def translation(self, write, extra: str = "", **keywords):
        with tempfile.TemporaryDirectory() as folder:
            return parse_parsec_input(write(Path(folder), extra, **keywords))

    def test_relaxation_with_a_move_flag_the_group_breaks(self) -> None:
        with patch.dict(os.environ, {name: "auto" for name in SWITCHES}):
            for extra, reason in ((_CHARGES_Z, "point_charges"), ("Foldblock_Side_Length: 2", "foldblock")):
                with self.subTest(reason=reason):
                    fixed_h = self.translation(_ch4_relax_input, extra, fixed="h")
                    self.assertEqual(fixed_h.symmetry_off_reasons, (reason,))
                    # Move_Flag fixes one H that an operation of the atoms exchanges with a movable one, also where
                    # the relaxation itself keeps the subgroup of the flags (PARSEC_MOVE_FLAG_SUBGROUP)
                    for subgroup in ("auto", "off"):
                        with patch.dict(os.environ, {"PARSEC_MOVE_FLAG_SUBGROUP": subgroup}):
                            self.assertEqual(module.kept_group_reasons(fixed_h), ())
                    # every atom movable, or the C alone fixed (an orbit of its own): the group is kept
                    self.assertEqual(module.kept_group_reasons(self.translation(_ch4_relax_input, extra, fixed="none")),
                                     (reason,))
                    self.assertEqual(module.kept_group_reasons(self.translation(_ch4_relax_input, extra, fixed="c")),
                                     (reason,))

    def test_spinors_keep_no_group(self) -> None:
        translation = self.translation(_au_soc_input, "Foldblock_Side_Length: 2")
        self.assertEqual(translation.symmetry_off_reasons, ("foldblock",))
        with patch.dict(os.environ, {module.FOLDBLOCK_SWITCH: "auto"}):
            self.assertEqual(module.kept_group_reasons(translation), ())

    def test_spin_channels_on_the_full_grid_keep_no_group(self) -> None:
        with patch.dict(os.environ, {module.FOLDBLOCK_SWITCH: "auto"}):
            spin = self.translation(_h2_input, "Foldblock_Side_Length: 2\nSpin_Polarization: .true.")
            self.assertEqual(module.kept_group_reasons(spin), ("foldblock",))
            full = self.translation(
                _h2_input, "Foldblock_Side_Length: 2\nSpin_Polarization: .true.\nSpin_Symmetry_Sectors: off"
            )
            self.assertEqual(module.kept_group_reasons(full), ())

    def test_fold_block_warnings(self) -> None:
        fold = self.translation(_h2_input, "Foldblock_Side_Length: 2")
        # the parser keeps its warning of Ignore_Symmetry (the bits of PARSEC_FOLDBLOCK_SYMMETRY=off)
        self.assertTrue(any(warning.startswith("Ignore_Symmetry=true reproduces") for warning in fold.warnings))
        self.assertEqual(module.kept_group_warnings(fold.warnings, ()), list(fold.warnings))
        self.assertEqual(module.kept_group_warnings(fold.warnings, ("polarizability",)), list(fold.warnings))
        with patch.dict(os.environ, {module.FOLDBLOCK_SWITCH: "auto"}):
            kept = module.kept_group_warnings(fold.warnings, ("foldblock",))
        self.assertFalse(any(warning.startswith("Ignore_Symmetry=true") for warning in kept))
        (switched,) = [warning for warning in kept if warning.startswith("Foldblock_Side_Length")]
        self.assertIn("symmetry is switched off, as in PARSEC 1.5", switched)
        self.assertIn("PARSEC_FOLDBLOCK_SYMMETRY=auto", switched)
        self.assertEqual(len(kept), len(fold.warnings) - 1)

    def test_kept_group_note_names_the_values(self) -> None:
        with patch.dict(os.environ, {name: "auto" for name in SWITCHES}):
            note = module.kept_group_note(("point_charges", "foldblock"))
        self.assertIn("PARSEC_POINT_CHARGE_SYMMETRY=auto", note)
        self.assertIn("PARSEC_FOLDBLOCK_SYMMETRY=auto", note)

    def run_dry(self, path: Path, environment: dict, *arguments: str) -> tuple[int, str]:
        from parsec_python.acceleration.cli import main

        stream = StringIO()
        with patch.dict(os.environ, environment), redirect_stdout(stream), redirect_stderr(stream):
            status = main([str(path), "--quiet", "--dry-run", *arguments])
        return status, stream.getvalue()

    def test_symmetry_off_on_the_command_line_keeps_no_group(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = _h2_input(Path(folder), "Polarizability: .true.")
            status, kept = self.run_dry(path, {module.POLARIZABILITY_SWITCH: "on"}, "--backend", "scipy")
            self.assertEqual(status, 0, kept)
            self.assertIn("Symmetry: mode=auto", kept)
            self.assertIn("PARSEC_POLARIZABILITY_SYMMETRY=on", kept)
            status, refused = self.run_dry(path, {module.POLARIZABILITY_SWITCH: "on"}, "--backend", "scipy",
                                           "--symmetry", "off")
            self.assertEqual(status, 0, refused)
            self.assertIn("Symmetry: mode=off", refused)
            self.assertNotIn("PARSEC 1.5 uses no point group here", refused)

    def test_restart_group_with_point_charges(self) -> None:
        """A density that keeps the group of the atoms and charges (D2d of CH4 with two charges on z) passes the
        test of a restart in their sectors and breaks that of the atoms alone (Td)."""

        grid = _grid()
        coordinates = np.asarray(grid.coordinates)
        density = np.exp(-(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2 + 0.5 * coordinates[:, 2] ** 2))
        charges = (PointCharge(0.3, [0.0, 0.0, 2.4]), PointCharge(0.3, [0.0, 0.0, -2.4]))
        charged = SimpleNamespace(grid=grid, atoms=TD, point_charges=charges)
        alone = SimpleNamespace(grid=grid, atoms=TD, point_charges=())
        self.assertIsNone(module.restart_density_breaking(charged, [density]))
        self.assertIsNotNone(module.restart_density_breaking(alone, [density]))
        # a density of Td passes both
        spherical = np.exp(-np.sum(coordinates**2, axis=1))
        self.assertIsNone(module.restart_density_breaking(alone, [spherical]))


def _grid(**options):
    return build_cluster_grid(GridSettings(spacing=0.5, radius=5.0, expansion_order=4, **options))


_A = 1.2
TD = (
    Atom("C", [0, 0, 0]),
    Atom("H", [_A, _A, _A]),
    Atom("H", [-_A, -_A, _A]),
    Atom("H", [-_A, _A, -_A]),
    Atom("H", [_A, -_A, -_A]),
)
C2H4 = (
    Atom("C", [0, 0, 0.67]),
    Atom("C", [0, 0, -0.67]),
    Atom("H", [0, 0.92, 1.23]),
    Atom("H", [0, -0.92, 1.23]),
    Atom("H", [0, 0.92, -1.23]),
    Atom("H", [0, -0.92, -1.23]),
)
H2O = (Atom("O", [0, 0, 0.12]), Atom("H", [0, 0.76, -0.48]), Atom("H", [0, -0.76, -0.48]))
H2 = (Atom("H", [0, 0, 0.7]), Atom("H", [0, 0, -0.7]))


class FieldPlanTests(unittest.TestCase):
    """Which of the six field loops are solved and which are images of a later one."""

    def plan(self, atoms, charges=(), **grid):
        return module.field_plan(_grid(**grid), atoms, atom_tolerance=1.0e-7, point_charges=charges)

    def test_plans(self) -> None:
        cases = (
            ("Td", TD, (), 24, (6,)),
            ("D2h", C2H4, (), 8, (2, 4, 6)),
            ("D4h", H2, (), 16, (4, 6)),
            ("C2v", H2O, (), 4, (2, 4, 5, 6)),
            ("H2 with a charge off the axes", H2, (PointCharge(0.3, [0.4, 2.0, 0.3]),), 1, (1, 2, 3, 4, 5, 6)),
        )
        for name, atoms, charges, order, solved in cases:
            with self.subTest(name=name):
                plan = self.plan(atoms, charges)
                self.assertEqual(len(plan.group), order)
                self.assertEqual(plan.solved, solved)
                self.assertEqual(set(plan.images) | set(plan.solved), {1, 2, 3, 4, 5, 6})
                # -z is always solved: its forces and state are those the run reports
                self.assertIn(6, plan.solved)
                self.assertEqual(plan.trivial, order == 1)

    def test_images_carry_the_field_and_the_dipole(self) -> None:
        from parsec_python.Energy.dipole import cluster_dipole_moment
        from parsec_python.SCF.polarizability import FIELD_DIRECTIONS

        grid = _grid()
        plan = module.field_plan(grid, C2H4, atom_tolerance=1.0e-7)
        pseudopotentials = {"C": SimpleNamespace(ionic_charge=4.0), "H": SimpleNamespace(ionic_charge=1.0)}
        coordinates = np.asarray(grid.coordinates)
        for image, (source, matrix, rows) in plan.images.items():
            with self.subTest(image=image):
                np.testing.assert_array_equal(matrix @ np.asarray(FIELD_DIRECTIONS[source]), FIELD_DIRECTIONS[image])
                self.assertEqual(np.sort(rows).tolist(), list(range(grid.size)))
                # a density polarized along the field of the source loop
                shift = 0.4 * np.asarray(FIELD_DIRECTIONS[source], dtype=float)
                density = np.exp(-np.sum((coordinates - shift) ** 2, axis=1))
                imaged = module._image_density(density, rows)
                expected = np.exp(-np.sum((coordinates - matrix @ shift) ** 2, axis=1))
                np.testing.assert_allclose(imaged, expected, atol=1e-14)
                before = cluster_dipole_moment(density, grid, C2H4, pseudopotentials).dipole
                after = cluster_dipole_moment(imaged, grid, C2H4, pseudopotentials).dipole
                np.testing.assert_allclose(after, matrix @ before, atol=1e-10)

    def test_stabilizers(self) -> None:
        plan = self.plan(TD)
        for axis in range(3):
            stabilizer = plan.stabilizer(axis)
            self.assertEqual(len(stabilizer), 4)
            unit = np.eye(3, dtype=np.int64)[axis]
            for matrix in stabilizer.values():
                np.testing.assert_array_equal(matrix @ unit, unit)

    def test_a_grid_that_breaks_an_operation_solves_its_loops(self) -> None:
        # A box longer along z: the permutations of z with x or y are no operations of the grid.
        grid = build_cluster_grid(
            GridSettings(spacing=0.5, radius=5.0, expansion_order=4, domain_shape="box", box_lengths=(8.0, 8.0, 10.0))
        )
        plan = module.field_plan(grid, TD, atom_tolerance=1.0e-7)
        self.assertEqual(len(plan.group), 24)
        self.assertIn(4, plan.solved)
        for image, (source, matrix, rows) in plan.images.items():
            self.assertIsNotNone(rows)


class ConstrainedDetectionTests(unittest.TestCase):
    """The detection with pseudo-atoms: the stabilizer of a field axis, the group of atoms and charges."""

    def detect(self, grid, atoms):
        from parsec_python.acceleration.Symmetry import (
            detect_point_group_reduction,
            load_or_detect_reflection_reduction,
        )

        abelian, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
        return abelian, detect_point_group_reduction(grid, atoms, abelian)

    def test_field_axis(self) -> None:
        grid = _grid()
        abelian, group = self.detect(grid, TD)
        self.assertEqual(int(group.group_order), 8)
        for axis in range(3):
            abelian, group = self.detect(grid, TD + (module.field_axis_constraint(axis),))
            self.assertEqual(int(abelian.group_order), 4)
            unit = np.eye(3)[axis]
            for operation in np.asarray(abelian.operations):
                np.testing.assert_array_equal(operation @ unit, unit)
        # H2 along z: the stabilizer of z is C4v, a weighted plan whose orbits are those of the two atoms
        constrained = H2 + (module.field_axis_constraint(2),)
        _, group = self.detect(grid, constrained)
        self.assertEqual(len(group.full_operations), 8)
        self.assertEqual(len(group.atom_representatives), 3)
        module.orbits_of_real_atoms(group, 2)
        self.assertEqual(group.atom_representatives.tolist(), [0, 1])
        self.assertEqual(group.atom_orbit_sizes.tolist(), [1, 1])

    def test_point_charges(self) -> None:
        grid = _grid()
        on_axis = module.point_charge_constraints((PointCharge(0.3, [0, 0, 2.4]), PointCharge(0.3, [0, 0, -2.4])))
        _, group = self.detect(grid, H2 + on_axis)
        self.assertEqual(len(group.full_operations), 16)
        unequal = module.point_charge_constraints((PointCharge(0.3, [0, 0, 2.4]), PointCharge(0.2, [0, 0, -2.4])))
        abelian, group = self.detect(grid, H2 + unequal)
        full = getattr(group, "full_operations", None)
        operations = np.asarray(abelian.operations if full is None else full).reshape(-1, 3, 3)
        self.assertGreater(len(operations), 1)
        for operation in operations:
            np.testing.assert_array_equal(operation @ np.array([0, 0, 1]), [0, 0, 1])
        off_axes = module.point_charge_constraints((PointCharge(0.3, [0.4, 2.0, 0.3]),))
        abelian, _ = self.detect(grid, H2 + off_axes)
        self.assertEqual(int(abelian.group_order), 1)

    def test_fold_blocks_without_shift(self) -> None:
        # PARSEC 1.5's grid of Foldblock_Side_Length 2 has no shift: the blocks [2m, 2m + 1] keep the permutations
        # of the axes, no sign change (a mirror x <-> y of the Td cluster here).
        grid = _grid(shift=(0.0, 0.0, 0.0), foldblock_side_length=2)
        abelian, group = self.detect(grid, TD)
        self.assertEqual(int(abelian.group_order), 2)
        for operation in np.asarray(abelian.operations):
            self.assertTrue(np.all(operation >= 0))


class PolarDatOrderTests(unittest.TestCase):
    """A field loop that solves out of the order of polar.f90 still writes polar.dat in that order."""

    def test_order_and_image_lines(self) -> None:
        from parsec_python.cli import _run_polarizability
        from parsec_python.Energy.dipole import DipoleMoment
        from parsec_python.SCF.polarizability import FIELD_DIRECTIONS, FieldRun, PolarizabilityResult

        def record(index):
            return FieldRun(
                direction=FIELD_DIRECTIONS[index],
                total_energy=-1.0 - 0.001 * index,
                dipole=DipoleMoment(charge=0.0, dipole=np.array([0.01 * index, 0.0, 0.0])),
                converged=True,
                iterations=0 if index in (1, 3, 5) else 4,
            )

        finished = []

        def loop(system, field, *, callback=None, before_run=None, after_run=None, **ignored):
            records = [record(index) for index in range(7)]
            for index in (0, 2, 1, 4, 3, 6, 5):
                if index in (1, 3, 5):
                    after_run(index, None, None, records[index], source=index + 1, operation=-np.eye(3))
                else:
                    before_run(index, FIELD_DIRECTIONS[index])
                    after_run(index, "system", f"result {index}", records[index])
            return PolarizabilityResult(field=field, runs=tuple(records), atom_count=2), None, "result 6"

        lines = []
        reporter = SimpleNamespace(finish=lambda result, elapsed, forces=None: finished.append(result), iteration=None)
        log = SimpleNamespace(write=lambda line="": lines.append(line))
        system = SimpleNamespace(atoms=H2, pseudopotentials={"H": None})
        translation = SimpleNamespace(polarizability_field=0.01, compute_forces=False)
        with tempfile.TemporaryDirectory() as folder:
            polarization, result, forces = _run_polarizability(
                system, translation, reporter, log, Path(folder), field_loop=loop, scf_runner=lambda *a, **k: None
            )
            text = (Path(folder) / "polar.dat").read_text()
        self.assertEqual(result, "result 6")
        self.assertEqual(finished, ["result 0", "result 2", "result 4", "result 6"])
        body = text.splitlines()
        start = next(index for index, line in enumerate(body) if "Component" in line) + 1
        rows = body[start : start + 7]
        # format 93 of polar.f90: the direction in 10 columns, then the energy
        energies = [float(row[10:31]) for row in rows]
        self.assertEqual(energies, sorted(energies, reverse=True))
        self.assertTrue(any("the image of run 3 (-x)" in line for line in lines), lines)


@dataclass(frozen=True)
class _FakeReference:
    grid: object
    atoms: tuple
    pseudopotentials: dict
    ionic_potential: np.ndarray
    point_charges: tuple = ()


class SymmetricFieldLoopTests(unittest.TestCase):
    """The loop of the field directions on a fake system: what is solved, on which group, from which start."""

    def system_of(self, grid, atoms, operations):
        from parsec_python.acceleration.SCF.single_point import AcceleratedPreparedSinglePointSystem

        reduction = SimpleNamespace(
            group_order=len(operations), operations=np.asarray(operations, dtype=np.int64), full_operations=None
        )
        reference = _FakeReference(
            grid=grid,
            atoms=atoms,
            pseudopotentials={
                "C": SimpleNamespace(ionic_charge=4.0),
                "H": SimpleNamespace(ionic_charge=1.0),
                "O": SimpleNamespace(ionic_charge=6.0),
            },
            ionic_potential=np.zeros(grid.size),
        )
        return AcceleratedPreparedSinglePointSystem(
            reference=reference,
            backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=SimpleNamespace(reduction=reduction))),
            backend_info=SimpleNamespace(details=(("symmetry_detection", f"{len(operations)} operations"),)),
            eigenproblem_solver=SimpleNamespace(start_new_scf_loop=lambda: "subspace", reset=lambda: None),
        )

    def run_loop(self, atoms):
        from parsec_python.SCF.polarizability import FIELD_DIRECTIONS

        grid = _grid()
        plan = module.field_plan(grid, atoms, atom_tolerance=1.0e-7)
        system = self.system_of(grid, atoms, list(plan.group.values()))
        coordinates = np.asarray(grid.coordinates)
        anisotropy = np.diag([1.0, 2.0, 3.0]) if len(plan.group) < 24 else np.eye(3)
        calls, prepared = [], []

        def fake_runner(system, *, iteration_state_callback=None):
            def run(reference, *, callback=None, warm_start=None):
                # the field direction from the ionic potential 2 F d.r
                potential = np.asarray(reference.ionic_potential)
                direction = np.rint(np.linalg.lstsq(coordinates, potential, rcond=None)[0] / 0.02).astype(int)
                group = len(system.backend.symmetry_eigensolver.decomposition.reduction.operations)
                calls.append((tuple(direction), group, None if warm_start is None else warm_start.eigensolver_state))
                density = np.exp(-np.sum((coordinates - 0.1 * direction) ** 2, axis=1))
                return SimpleNamespace(
                    energies=SimpleNamespace(total=-1.0 - 0.01 * float(direction @ anisotropy @ direction)),
                    density=density,
                    converged=True,
                    iterations=3,
                    input_effective_potential=potential,
                    hartree_potential=np.zeros_like(potential),
                    spin=None,
                )

            return run

        def fake_prepare(problem, *, backend, symmetry, symmetry_cache_directory, symmetry_constraints=()):
            (constraint,) = symmetry_constraints
            axis = int(np.argmax(np.abs(constraint.position)))
            prepared.append(axis)
            # the mode of the run, never one it did not ask for
            self.assertEqual(symmetry, "auto")
            return self.system_of(grid, atoms, list(plan.stabilizer(axis).values()))

        records = {}
        self.lines = []
        with patch("parsec_python.acceleration.SCF.polarizability.field_runner", fake_runner), patch(
            "parsec_python.acceleration.driver.prepare_single_point", fake_prepare
        ), patch.dict(os.environ, {module.POLARIZABILITY_SWITCH: "on"}):
            loop = module.symmetric_field_loop(
                system,
                problem=None,
                backend="cupy",
                symmetry_cache_directory=None,
                log=SimpleNamespace(write=self.lines.append),
            )
            result, _, last = loop(
                system.reference,
                0.01,
                before_run=lambda index, direction: None,
                after_run=lambda index, run_system, result, record, **kw: records.setdefault(index, (result, record)),
            )
        # every loop has its record, the energies and dipoles of a direct solve of each direction
        self.assertEqual(sorted(records), list(range(7)))
        for index, run in enumerate(result.runs):
            direction = np.asarray(FIELD_DIRECTIONS[index])
            self.assertAlmostEqual(run.total_energy, -1.0 - 0.01 * float(direction @ anisotropy @ direction), places=12)
        self.assertEqual(calls[-1][0], (0, 0, -1))
        self.assertIs(last, records[6][0])
        return plan, calls, prepared, records

    def test_td(self) -> None:
        plan, calls, prepared, records = self.run_loop(TD)
        # the zero field in G (24 on this fake system), -z in the 4 operations that keep z, from the zero-field
        # potential without a subspace
        self.assertEqual([call[:2] for call in calls], [((0, 0, 0), 24), ((0, 0, -1), 4)])
        self.assertEqual(calls[1][2], None)
        self.assertEqual(prepared, [2])
        self.assertEqual([index for index, (result, _) in records.items() if result is None], [1, 2, 3, 4, 5])
        # the line of the images names the value of the switch
        (images,) = [line for line in self.lines if "image of" in line]
        self.assertIn("PARSEC_POLARIZABILITY_SYMMETRY=on", images)

    def test_a_run_without_symmetry_takes_no_field_loop(self) -> None:
        grid = _grid()
        system = self.system_of(grid, TD, [np.eye(3)])
        with self.assertRaises(ValueError):
            module.symmetric_field_loop(system, problem=None, backend="cupy", symmetry_cache_directory=None, symmetry="off")

    def test_d2h(self) -> None:
        plan, calls, prepared, records = self.run_loop(C2H4)
        self.assertEqual([call[0] for call in calls], [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)])
        self.assertEqual(prepared, [0, 1, 2])

    def test_c2v_axis_loops_follow_each_other(self) -> None:
        plan, calls, prepared, records = self.run_loop(H2O)
        self.assertEqual([call[0] for call in calls], [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        # -z continues the subspace of +z, as polar.f90
        self.assertEqual(calls[-1][2], "subspace")
        self.assertEqual(calls[-2][2], None)

    def test_a_spin_loop_that_leaves_the_sectors_keeps_its_field(self) -> None:
        """The guard of the spin sectors sends the -x loop of a D2h molecule to the full grid: the full grid of
        the fallback, prepared from the input without the field, takes the field of the loop, and the loops that
        follow solve there without another preparation."""

        from parsec_python.SCF.polarizability import FIELD_DIRECTIONS

        grid = _grid()
        atoms = C2H4
        plan = module.field_plan(grid, atoms, atom_tolerance=1.0e-7)
        coordinates = np.asarray(grid.coordinates)
        anisotropy = np.diag([1.0, 2.0, 3.0])
        calls, prepared, fallbacks = [], [], []

        def direction_of(reference):
            potential = np.asarray(reference.ionic_potential)
            return tuple(np.rint(np.linalg.lstsq(coordinates, potential, rcond=None)[0] / 0.02).astype(int))

        def solve(system, *, callback=None, warm_start=None, **keywords):
            direction = direction_of(system.reference)
            group = len(system.backend.symmetry_eigensolver.decomposition.reduction.operations)
            calls.append((direction, group))
            vector = np.asarray(direction, dtype=float)
            density = np.exp(-np.sum((coordinates - 0.1 * vector) ** 2, axis=1))
            potential = np.asarray(system.reference.ionic_potential)
            return SimpleNamespace(
                energies=SimpleNamespace(total=-1.0 - 0.01 * float(vector @ anisotropy @ vector)),
                density=density,
                converged=True,
                iterations=3,
                input_effective_potential=potential,
                hartree_potential=np.zeros_like(potential),
                spin=None,
            )

        test = self

        class Fallback:
            """Leaves the sectors in the loop along -x, as _SpinSectorFallback: the full grid from the input."""

            def run(self, system, run, *, callback=None, warm_start=None, **keywords):
                if direction_of(system.reference) == (-1, 0, 0):
                    self.replace_with_full_grid(system, "the SCF loop", None)
                return run(system, callback=callback, warm_start=warm_start, **keywords)

            def replace_with_full_grid(self, system, stage, report):
                fallbacks.append(stage)
                full_grid = test.system_of(grid, atoms, [np.eye(3)])
                for name in ("reference", "backend", "backend_info", "eigenproblem_solver", "spin_sector_fallback"):
                    setattr(system, name, getattr(full_grid, name))

        def fake_run_scf(system, *, callback=None, warm_start=None, **keywords):
            fallback = system.spin_sector_fallback
            if fallback is None:
                return solve(system, callback=callback, warm_start=warm_start, **keywords)
            return fallback.run(system, solve, callback=callback, warm_start=warm_start, **keywords)

        def fake_prepare(problem, *, backend, symmetry, symmetry_cache_directory, symmetry_constraints=()):
            (constraint,) = symmetry_constraints
            axis = int(np.argmax(np.abs(constraint.position)))
            prepared.append(axis)
            stabilized = self.system_of(grid, atoms, list(plan.stabilizer(axis).values()))
            stabilized.spin_sector_fallback = Fallback()
            return stabilized

        system = self.system_of(grid, atoms, list(plan.group.values()))
        system.spin_sector_fallback = Fallback()
        lines = []
        with patch("parsec_python.acceleration.driver.run_scf", fake_run_scf), patch(
            "parsec_python.acceleration.driver.prepare_single_point", fake_prepare
        ), patch.object(module, "_free_device_pools", lambda: None):
            loop = module.symmetric_field_loop(
                system,
                problem=None,
                backend="cupy",
                symmetry_cache_directory=None,
                log=SimpleNamespace(write=lines.append),
            )
            result, last_system, _ = loop(system.reference, 0.01)
        self.assertEqual(fallbacks, ["the SCF loop"])
        # -x prepared in its stabilizer, then the full grid (with the field of -x) and no other preparation
        self.assertEqual(prepared, [0])
        self.assertEqual(calls, [((0, 0, 0), 8), ((-1, 0, 0), 1), ((0, -1, 0), 1), ((0, 0, -1), 1)])
        for index, run in enumerate(result.runs):
            direction = np.asarray(FIELD_DIRECTIONS[index])
            self.assertAlmostEqual(run.total_energy, -1.0 - 0.01 * float(direction @ anisotropy @ direction), places=12)
        # the system holds the full grid without a field; the last loop's system carries the field of -z
        self.assertIsNone(system.spin_sector_fallback)
        self.assertFalse(np.any(np.asarray(system.reference.ionic_potential)))
        self.assertEqual(direction_of(last_system), (0, 0, -1))
        self.assertTrue(any("left the spin sectors" in line for line in lines))


def _run_cli(path: Path, environment: dict, *arguments: str) -> tuple[int, str]:
    from parsec_python.acceleration.cli import main

    stream = StringIO()
    with patch.dict(os.environ, environment), redirect_stdout(stream), redirect_stderr(stream):
        status = main([str(path), "--quiet", *arguments])
    return status, stream.getvalue()


def _archive(directory: Path) -> dict:
    with np.load(directory / "parsec_python_results.npz") as archive:
        return {name: archive[name] for name in archive.files}


@unittest.skipUnless(GPU, "CuPy, a usable GPU and the native extension are required")
class AcceleratedRunTests(unittest.TestCase):
    """Complete runs of the accelerated CLI under ``auto`` against ``off``, the full grid of before."""

    def pair(self, write, extra: str, switch: str, *, setup=None, first: str = "auto"):
        """``(first, off)``: the archive, parsec.out and the directory of each run."""

        runs = []
        for value in (first, "off"):
            directory = Path(tempfile.mkdtemp(prefix=f"parsec_symoff_{value}_"))
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            path = write(directory, extra)
            if setup is not None:
                setup(directory, path)
            status, output = _run_cli(path, {switch: value})
            self.assertIn(status, (0, 3), output)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            runs.append((_archive(directory), log, directory))
        return runs

    def assert_same(self, auto, off, *, energy=1.0e-7) -> None:
        self.assertAlmostEqual(float(auto["energy_total_ry"]), float(off["energy_total_ry"]), delta=energy)

    @staticmethod
    def polar_rows(directory: Path) -> np.ndarray:
        # format 93 of polar.f90: the direction (10 columns), the energy (F17.12), the dipole (3 F11.6)
        body = (directory / "polar.dat").read_text().splitlines()
        start = next(index for index, line in enumerate(body) if "Component" in line) + 1
        rows = []
        for line in body[start : start + 7]:
            rows.append([float(line[10:31])] + [float(line[31 + 11 * k : 42 + 11 * k]) for k in range(3)])
        return np.asarray(rows)

    def check_polarizability(self, write, extra: str, *, forces: bool = False) -> None:
        # on: the sectors whatever the size (auto keeps these small inputs on the full grid)
        (auto, auto_log, auto_dir), (off, off_log, off_dir) = self.pair(
            write, extra, module.POLARIZABILITY_SWITCH, first="on"
        )
        np.testing.assert_allclose(
            auto["polarizability_energies_ry"], off["polarizability_energies_ry"], rtol=0.0, atol=2.0e-7
        )
        np.testing.assert_allclose(
            auto["polarizability_from_dipole_ang3"], off["polarizability_from_dipole_ang3"], rtol=2.0e-3, atol=1.0e-4
        )
        np.testing.assert_allclose(
            auto["polarizability_from_energy_ang3"], off["polarizability_from_energy_ang3"], rtol=2.0e-2, atol=2.0e-3
        )
        auto_rows, off_rows = self.polar_rows(auto_dir), self.polar_rows(off_dir)
        self.assertEqual(auto_rows.shape, (7, 4))
        np.testing.assert_allclose(auto_rows[:, 0], off_rows[:, 0], atol=2.0e-7)
        # dipoles in Debye
        np.testing.assert_allclose(auto_rows[:, 1:], off_rows[:, 1:], atol=2.0e-4)
        self.assertIn("not solved", auto_log)
        self.assertNotIn("not solved", off_log)
        self.assertIn("PARSEC_POLARIZABILITY_SYMMETRY=on", auto_log)
        if forces:
            np.testing.assert_allclose(auto["forces_total"], off["forces_total"], atol=1.0e-6)

    def test_polarizability_td(self) -> None:
        self.check_polarizability(_ch4_input, "Polarizability: .true.\nElectric_Field: 0.005")

    def test_polarizability_h2_with_forces(self) -> None:
        self.check_polarizability(
            _h2_input, "Polarizability: .true.\nElectric_Field: 0.01\nSkip_Force: .false.", forces=True
        )

    def test_point_charges(self) -> None:
        symmetric = (
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\n"
            "begin Point_Coord\n 0.0 0.0 2.4\n 0.0 0.0 -2.4\nend Point_Coord\nSkip_Force: .false."
        )
        (auto, auto_log, _), (off, off_log, _) = self.pair(_h2_input, symmetric, module.POINT_CHARGE_SWITCH)
        self.assert_same(auto, off)
        np.testing.assert_allclose(auto["forces_total"], off["forces_total"], atol=1.0e-6)
        self.assertIn("PARSEC_POINT_CHARGE_SYMMETRY=auto", auto_log)
        self.assertIn("sectors", auto_log.lower())
        # parsec.out: the symmetry block of PARSEC 1.5 for the input (no Ignore_Symmetry), as before under off
        self.assertNotIn("Symmetry suppressed by the Ignore_Symmetry flag", auto_log)
        self.assertIn(" ignoresym= F", auto_log)
        self.assertIn("Symmetry suppressed by the Ignore_Symmetry flag", off_log)
        asymmetric = (
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\n"
            "begin Point_Coord\n 0.4 2.0 0.3\nend Point_Coord"
        )
        (auto, _, _), (off, _, _) = self.pair(_h2_input, asymmetric, module.POINT_CHARGE_SWITCH)
        self.assert_same(auto, off)

    def test_fold_blocks(self) -> None:
        (auto, auto_log, _), (off, off_log, _) = self.pair(
            _ch4_input, "Foldblock_Side_Length: 2", module.FOLDBLOCK_SWITCH
        )
        self.assert_same(auto, off)
        self.assertIn("PARSEC_FOLDBLOCK_SYMMETRY=auto", auto_log)
        # the warning of Ignore_Symmetry stays under off (as before) and goes under auto, where the warning of the
        # fold blocks names the group kept
        self.assertIn("Ignore_Symmetry=true reproduces", off_log)
        self.assertNotIn("Ignore_Symmetry=true reproduces", auto_log)
        self.assertIn("the accelerated driver keeps the operations", auto_log)
        self.assertNotIn("the accelerated driver keeps the operations", off_log)

    def relaxation_pair(self, write, extra: str, switch: str, **keywords):
        runs = []
        for value in ("auto", "off"):
            directory = Path(tempfile.mkdtemp(prefix=f"parsec_symoff_relax_{value}_"))
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            status, output = _run_cli(write(directory, extra, **keywords), {switch: value})
            # 4: the relaxation stopped at Movement_Num before Force_Min
            self.assertIn(status, (0, 3, 4), output)
            runs.append((_archive(directory), (directory / "parsec.out").read_text(encoding="utf-8")))
        return runs

    def test_relaxation_with_a_move_flag_the_group_breaks(self) -> None:
        """BFGS with Move_Flag fixing one H of CH4, which the kept group (D2d with charges on z, the axis swaps of
        the fold blocks) exchanges with movable ones: the full grid of off, the energies of off (before 2026-11-19
        the run stopped with 'the atoms do not have the symmetry of the sectors')."""

        for extra, switch in ((_CHARGES_Z, module.POINT_CHARGE_SWITCH), ("Foldblock_Side_Length: 2", module.FOLDBLOCK_SWITCH)):
            with self.subTest(switch=switch):
                (auto, auto_log), (off, off_log) = self.relaxation_pair(_ch4_relax_input, extra, switch, fixed="h")
                self.assertNotIn("PARSEC 1.5 uses no point group here", auto_log)
                self.assertIn("orbital_symmetry = full grid", auto_log)
                self.assertAlmostEqual(float(auto["energy_total_ry"]), float(off["energy_total_ry"]), delta=1.0e-10)
                self.assertEqual(set(auto), set(off))
                if "forces_total" in auto:
                    np.testing.assert_allclose(auto["forces_total"], off["forces_total"], atol=1.0e-10)

    def test_spin_orbit_with_fold_blocks(self) -> None:
        """Self-consistent spin-orbit coupling (the spinor driver takes no sectors) with Foldblock_Side_Length 2:
        the full grid under auto, the energy of off (before 2026-11-19: a TypeError in the spinor driver)."""

        archives = {}
        for value in ("auto", "off"):
            directory = Path(tempfile.mkdtemp(prefix=f"parsec_symoff_soc_{value}_"))
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            status, output = _run_cli(
                _au_soc_input(directory, "Foldblock_Side_Length: 2"), {module.FOLDBLOCK_SWITCH: value}
            )
            self.assertEqual(status, 0, output)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            self.assertNotIn("PARSEC 1.5 uses no point group here", log)
            archives[value] = _archive(directory)
        self.assertAlmostEqual(
            float(archives["auto"]["energy_total_ry"]), float(archives["off"]["energy_total_ry"]), delta=1.0e-10
        )

    def test_symmetry_off_on_the_command_line(self) -> None:
        """--symmetry off with PARSEC_POLARIZABILITY_SYMMETRY=on: the seven loops on the full grid, as under off."""

        archives, logs = {}, {}
        for value in ("on", "off"):
            directory = Path(tempfile.mkdtemp(prefix=f"parsec_symoff_flag_{value}_"))
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            path = _h2_input(directory, "Polarizability: .true.\nElectric_Field: 0.01")
            status, output = _run_cli(path, {module.POLARIZABILITY_SWITCH: value}, "--symmetry", "off")
            self.assertIn(status, (0, 3), output)
            archives[value] = _archive(directory)
            logs[value] = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertNotIn("not solved", logs["on"])
        self.assertNotIn("prepared again", logs["on"])
        self.assertNotIn("PARSEC 1.5 uses no point group here", logs["on"])
        np.testing.assert_allclose(
            archives["on"]["polarizability_energies_ry"], archives["off"]["polarizability_energies_ry"], rtol=0.0, atol=1.0e-12
        )

    def test_restart_from_wfndat(self) -> None:
        from parsec_python.Output.wfn_dat import wfn_dat_from_result, write_wfn_dat
        from parsec_python.SCF import prepare_single_point, run_scf

        def setup(directory, path):
            translation = parse_parsec_input(_h2_input(directory, criterion="1.0e-8"))
            first = run_scf(prepare_single_point(translation.problem))
            write_wfn_dat(directory / "wfn.dat", wfn_dat_from_result(first, first.grid))
            _h2_input(directory, "Restart_Run: .true.\nRestart_From_Wfndat: .true.")
            self.energy = first.energies.total

        (auto, auto_log, _), (off, _, _) = self.pair(_h2_input, "", module.WFNDAT_RESTART_SWITCH, setup=setup)
        self.assertIn("keep the point group of the atoms", auto_log)
        self.assertAlmostEqual(float(auto["energy_total_ry"]), self.energy, delta=1.0e-6)
        self.assert_same(auto, off, energy=1.0e-6)


@unittest.skipUnless(GPU, "CuPy, a usable GPU and the native extension are required")
class SpinRestartTests(unittest.TestCase):
    """A spin-polarized Restart_Run of O2 keeps the sectors where parsec.dat keeps the group of the atoms."""

    def write(self, directory: Path, extra=()) -> Path:
        from parsec_python.tests.spin_reference import load_cases, write_case

        case = dict(load_cases()["o2_ca"])
        return write_case(case, directory, extra=tuple(extra))

    def first_run(self, directory: Path, extra=()) -> None:
        path = self.write(directory, extra)
        status, output = _run_cli(path, {})
        self.assertIn(status, (0, 3), output)

    def restart(self, directory: Path, value: str, extra=()) -> tuple[dict, str]:
        path = self.write(directory, ("Restart_Run: .true.", *extra))
        status, output = _run_cli(path, {module.SPIN_RESTART_SWITCH: value})
        self.assertIn(status, (0, 3), output)
        return _archive(directory), (directory / "parsec.out").read_text(encoding="utf-8")

    def copies(self, count: int = 2):
        folders = [Path(tempfile.mkdtemp(prefix="parsec_spinrestart_")) for _ in range(count)]
        for folder in folders:
            self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return folders

    def test_a_symmetric_restart_stays_in_the_sectors(self) -> None:
        # on: the sectors whatever the size; auto keeps this small O2 on the full grid of off
        first, second, third = self.copies(3)
        self.first_run(first)
        shutil.copyfile(first / "parsec.dat", second / "parsec.dat")
        shutil.copyfile(first / "parsec.dat", third / "parsec.dat")
        energy = float(_archive(first)["energy_total_ry"])
        on, on_log = self.restart(first, "on")
        off, off_log = self.restart(second, "off")
        auto, auto_log = self.restart(third, "auto")
        self.assertIn("keep the point group of the atoms", on_log)
        self.assertIn("PARSEC_SPIN_RESTART_SECTORS=on", on_log)
        self.assertIn("both channels on the full grid", off_log)
        self.assertIn("both channels on the full grid", auto_log)
        self.assertNotIn("keep the point group of the atoms", auto_log)
        self.assertAlmostEqual(float(on["energy_total_ry"]), energy, delta=1.0e-7)
        self.assertAlmostEqual(float(on["energy_total_ry"]), float(off["energy_total_ry"]), delta=1.0e-7)
        # auto below the size rule: the run of off
        self.assertEqual(int(auto["iterations"]), int(off["iterations"]))
        self.assertAlmostEqual(float(auto["energy_total_ry"]), float(off["energy_total_ry"]), delta=1.0e-12)

    def test_a_restart_with_point_charges_tests_their_group(self) -> None:
        """O2 with one charge on its axis (C4v of the D4h of the atoms): the stored densities keep C4v, the group
        the restart was prepared in, and the restart stays in its sectors (before 2026-11-19 it tested D4h of the
        atoms alone and was prepared again on the full grid)."""

        charge = ("Add_Point_Charges: .true.", "Point_Typ_Num: 1", "Pt_Chg: 0.2", "begin Point_Coord",
                  " 0.0 0.0 3.0", "end Point_Coord")
        first, second = self.copies()
        self.first_run(first, charge)
        shutil.copyfile(first / "parsec.dat", second / "parsec.dat")
        energy = float(_archive(first)["energy_total_ry"])
        on, on_log = self.restart(first, "on", charge)
        off, off_log = self.restart(second, "off", charge)
        self.assertIn("keep the point group of the atoms and the point charges", on_log)
        self.assertNotIn("prepared again on the full grid", on_log)
        self.assertIn("both channels on the full grid", off_log)
        self.assertAlmostEqual(float(on["energy_total_ry"]), energy, delta=2.0e-7)
        self.assertAlmostEqual(float(on["energy_total_ry"]), float(off["energy_total_ry"]), delta=2.0e-7)

    def test_spin_polarizability(self) -> None:
        """The seven loops of the O2 triplet in the sectors of each field axis, against the full grid."""

        archives = {}
        for value in ("on", "off"):
            (folder,) = self.copies()[:1]
            path = self.write(folder, ("Polarizability: .true.", "Electric_Field: 0.005"))
            status, output = _run_cli(path, {module.POLARIZABILITY_SWITCH: value})
            self.assertIn(status, (0, 3), output)
            archives[value] = _archive(folder)
            if value == "on":
                text = (folder / "parsec.out").read_text(encoding="utf-8")
                self.assertIn("not solved", text)
                # the guard of the spin sectors tests the group of the atoms and the field axis, which the
                # densities keep: no loop leaves the sectors
                self.assertNotIn("NOTE: Spin_Symmetry_Sectors:", text)
                self.assertNotIn("left the spin sectors", text)
        np.testing.assert_allclose(
            archives["on"]["polarizability_energies_ry"],
            archives["off"]["polarizability_energies_ry"],
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            archives["on"]["polarizability_from_dipole_ang3"],
            archives["off"]["polarizability_from_dipole_ang3"],
            rtol=2.0e-3,
        )

    def test_spin_point_charge(self) -> None:
        """One charge on the axis of the O2 triplet keeps C4v of its D4h: the two channels stay in the sectors
        of C4v (the guard tests the atoms with the charge), with the energy of the full grid."""

        archives, texts = {}, {}
        charge = ("Add_Point_Charges: .true.", "Point_Typ_Num: 1", "Pt_Chg: 0.2", "begin Point_Coord",
                  " 0.0 0.0 3.0", "end Point_Coord")
        for value in ("auto", "off"):
            (folder,) = self.copies()[:1]
            path = self.write(folder, charge)
            status, output = _run_cli(path, {module.POINT_CHARGE_SWITCH: value})
            self.assertIn(status, (0, 3), output)
            archives[value] = _archive(folder)
            texts[value] = (folder / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("PARSEC_POINT_CHARGE_SYMMETRY=auto", texts["auto"])
        self.assertNotIn("NOTE: Spin_Symmetry_Sectors:", texts["auto"])
        self.assertAlmostEqual(
            float(archives["auto"]["energy_total_ry"]), float(archives["off"]["energy_total_ry"]), delta=2.0e-7
        )

    def test_a_broken_restart_goes_to_the_full_grid(self) -> None:
        from dataclasses import replace

        from parsec_python.Output.parsec_dat import read_parsec_dat, write_parsec_dat

        first, second = self.copies()
        # A full-grid run, whose parsec.dat holds every point; its up density is then made asymmetric in x.
        self.first_run(first, ("Spin_Symmetry_Sectors: off",))
        data = read_parsec_dat(first / "parsec.dat")
        points = np.asarray(data.points, dtype=float)
        weight = 1.0 + 0.3 * np.tanh(points[:, 0])
        densities = (np.asarray(data.densities[0]) * weight,) + tuple(data.densities[1:])
        write_parsec_dat(first / "parsec.dat", replace(data, densities=densities))
        shutil.copyfile(first / "parsec.dat", second / "parsec.dat")
        auto, auto_log = self.restart(first, "on")
        off, _ = self.restart(second, "off")
        self.assertIn("break the point group of the atoms", auto_log)
        # prepared again as off prepares it, from the same file: the same run
        self.assertEqual(int(auto["iterations"]), int(off["iterations"]))
        self.assertAlmostEqual(float(auto["energy_total_ry"]), float(off["energy_total_ry"]), delta=1.0e-10)


if __name__ == "__main__":
    unittest.main()
