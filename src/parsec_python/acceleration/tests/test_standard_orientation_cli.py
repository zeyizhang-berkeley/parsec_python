"""The command line with ``PARSEC_STANDARD_ORIENTATION=on`` (:mod:`parsec_python.Input.standard_orientation`).

A rotated, displaced methane with the switch on is calculated in the frame of its group Td (weighted sectors of
D2d), which is the frame of the aligned methane: the energy is that of the aligned input without the switch, and the
forces and the structures of a relaxation that parsec.out, ``atom.cor`` and ``relaxation.xyz`` report are those of
the aligned run turned into the frame of the input.  The aligned input keeps its bits with the switch on.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.acceleration.backends.cupy import cupy_available

EXAMPLE = Path(__file__).resolve().parents[3].parent / "examples" / "0d_benzene"
_A = 1.18 * 1.05
METHANE = np.array([[0.0, 0.0, 0.0], [_A, _A, _A], [-_A, -_A, _A], [-_A, _A, -_A], [_A, -_A, -_A]])


def _rotation(z: float, y: float, x: float) -> np.ndarray:
    def axis(angle, first, second):
        cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        matrix = np.eye(3)
        matrix[first, first] = matrix[second, second] = cosine
        matrix[first, second], matrix[second, first] = -sine, sine
        return matrix

    return axis(z, 0, 1) @ axis(y, 2, 0) @ axis(x, 1, 2)


ROTATION = _rotation(10.0, 0.0, 0.0)


def _write(directory: Path, positions: np.ndarray, extra: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("C_POTRE.DAT", "H_POTRE.DAT"):
        shutil.copy(EXAMPLE / name, directory / name)
    rows = ["  " + "  ".join(f"{value:.14f}" for value in row) for row in positions]
    text = "\n".join(
        [
            "Boundary_Sphere_Radius: 7.0",
            "Grid_Spacing: 0.35",
            "States_Num: 6",
            "Convergence_Criterion: 1.0e-7",
            "Atom_Types_Num: 2",
            "Atom_Type: C",
            "Local_Component: p",
            "begin Atom_Coord",
            rows[0],
            "end Atom_Coord",
            "Atom_Type: H",
            "Local_Component: s",
            "begin Atom_Coord",
            *rows[1:],
            "end Atom_Coord",
            extra,
        ]
    )
    path = directory / "parsec.in"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _final_forces(text: str) -> np.ndarray:
    block = text.split("coordinates and tot. forces (after setting net force to zero)")[-1]
    rows = []
    for line in block.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 7 and parts[0].isdigit():
            rows.append([float(value) for value in parts[1:]])
        elif rows:
            break
    return np.asarray(rows)


@unittest.skipUnless(cupy_available(), "CuPy GPU is not available")
class StandardOrientationCommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def _run(
        self, name: str, positions: np.ndarray, extra: str, switch: str, statuses: tuple[int, ...] = (0,)
    ) -> tuple[str, Path]:
        # a relaxation that stops at Movement_Num before Force_Min exits with 4 (relaxation_exit_code)
        path = _write(self.directory / name, positions, extra)
        with mock.patch.dict(os.environ, {"PARSEC_STANDARD_ORIENTATION": switch}), redirect_stdout(
            StringIO()
        ), redirect_stderr(StringIO()) as errors:
            status = cli_module.main([str(path), "--quiet", "--no-archive", "--no-symmetry-cache"])
        self.assertIn(status, statuses, errors.getvalue())
        return (path.parent / "parsec.out").read_text(encoding="utf-8"), path.parent

    @staticmethod
    def _energy(text: str) -> float:
        return float(re.findall(r"Total Energy =\s+(-?\d+\.\d+)", text)[-1])

    def test_single_point_with_forces(self) -> None:
        forces = "Calculate_Force: .true."
        aligned, _ = self._run("aligned_off", METHANE, forces, "off")
        kept, _ = self._run("aligned_on", METHANE, forces, "on")
        rotated, _ = self._run("rotated_on", METHANE @ ROTATION.T, forces, "on")
        self.assertIn("frame of the input keeps the largest group", kept)
        self.assertIn("Standard orientation (PARSEC_STANDARD_ORIENTATION=on)", rotated)
        # the aligned input keeps its bits; the rotated one is calculated in the aligned frame
        self.assertEqual(self._energy(kept), self._energy(aligned))
        self.assertAlmostEqual(self._energy(rotated), self._energy(aligned), delta=1.0e-7)
        reference = _final_forces(aligned)
        reported = _final_forces(rotated)
        np.testing.assert_allclose(reported[:, :3], reference[:, :3] @ ROTATION.T, atol=2.0e-6)
        np.testing.assert_allclose(reported[:, 3:], reference[:, 3:] @ ROTATION.T, atol=2.0e-6)
        self.assertGreater(float(np.max(np.abs(reference[:, 3:]))), 1.0e-3)

    def test_relaxation_files_are_in_the_frame_of_the_input(self) -> None:
        from parsec_python.Relaxation.files import parse_xyz_frames

        relax = "Minimization: BFGS\nMovement_Num: 2\nForce_Min: 1.0e-4"
        _, aligned = self._run("relax_off", METHANE, relax, "off", statuses=(0, 4))
        text, rotated = self._run("relax_on", METHANE @ ROTATION.T, relax, "on", statuses=(0, 4))
        frames = parse_xyz_frames((aligned / "relaxation.xyz").read_text())
        turned = parse_xyz_frames((rotated / "relaxation.xyz").read_text())
        self.assertEqual(len(frames), len(turned))
        self.assertGreaterEqual(len(frames), 2)
        for (_, first), (_, second) in zip(frames, turned):
            np.testing.assert_allclose(np.asarray(second), np.asarray(first) @ ROTATION.T, atol=2.0e-6)
        self.assertIn("Standard orientation", text)

    def test_relax_restart_continues_in_the_frame_of_the_first_run(self) -> None:
        # off the origin by more than 0.1 bohr: the first run recentres the input, a Restart_Run does not; both
        # restarts report in the frame of the first run, which relax_restart.dat records
        from parsec_python.Input.standard_orientation import StandardOrientation
        from parsec_python.Relaxation.files import ORIENTATION_HEADER, parse_relax_restart, parse_xyz_frames

        positions = METHANE @ ROTATION.T + np.array([0.7, -0.4, 0.3])
        relax = "Minimization: BFGS\nMovement_Num: 2\nForce_Min: 1.0e-5\nWrite_Relax_Restart: .true."
        _, first = self._run("first", positions, relax, "on", statuses=(0, 4))
        text = (first / "relax_restart.dat").read_text(encoding="utf-8")
        self.assertIn(ORIENTATION_HEADER, text)
        restart = parse_relax_restart(text, 5)
        expected = StandardOrientation.from_restart_records(restart.orientation).to_reference(restart.positions)
        for name, extra in (("restart", ""), ("restart_run", "\nRestart_Run: .true.")):
            with self.subTest(name=name):
                directory = self.directory / name
                shutil.copytree(first, directory)
                for stale in ("relaxation.xyz", "atom.cor", "parsec.out"):
                    (directory / stale).unlink(missing_ok=True)
                again = relax.replace("Movement_Num: 2", "Movement_Num: 1") + "\nRelax_Restart: .true." + extra
                log, _ = self._run(name, positions, again, "on", statuses=(0, 4))
                self.assertIn("calculation frame it records", log)
                frames = parse_xyz_frames((directory / "relaxation.xyz").read_text())
                np.testing.assert_allclose(np.asarray(frames[0][1]), expected, atol=1.0e-8)
        # the file needs the switch on
        directory = self.directory / "restart_off"
        shutil.copytree(first, directory)
        path = _write(directory, positions, relax + "\nRelax_Restart: .true.")
        with mock.patch.dict(os.environ, {"PARSEC_STANDARD_ORIENTATION": "off"}), redirect_stdout(
            StringIO()
        ), redirect_stderr(StringIO()) as errors:
            status = cli_module.main([str(path), "--quiet", "--no-archive", "--no-symmetry-cache"])
        self.assertEqual(status, 2)
        self.assertIn("PARSEC_STANDARD_ORIENTATION=on", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
