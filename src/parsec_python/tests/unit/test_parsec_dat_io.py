"""parsec.dat, eigen.dat and cube files in the layouts of PARSEC.

The byte layout of ``parsec.dat`` is checked against ``scipy.io.FortranFile``
(gfortran sequential records); the text layouts against lines that the
PARSEC executable wrote for an H2O cluster (``eigensave.f90`` and
``export_grid_data.f90``, Ignore_Symmetry, 0.57 bohr, 10.4 bohr sphere).
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import FortranFile

from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.models import (
    GridSettings,
    OutputFileSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SpinChannelResult,
    WavefunctionSelection,
)
from parsec_python.Output import parsec_dat as dat
from parsec_python.Output.cube import cube_text, dense_box, fortran_exponential
from parsec_python.Output.eigen_dat import (
    EigenChannel,
    eigen_dat_block,
    eigen_dat_header,
    eigen_dat_line,
)
from parsec_python.Output.output_files import RunOutputFiles, output_header_lines
from parsec_python.SCF.restart import (
    RestartError,
    check_restart_file,
    restart_fields,
    restart_warm_start,
    with_first_warm_start,
)


def _cluster_grid(spacing=0.8, radius=3.0, shift=(0.5, 0.5, 0.5)):
    return build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=4, shift=shift))


class _Result:
    """The fields of a scalar SinglePointResult that parsec.dat reads."""

    def __init__(self, grid, states=4, occupied=2, rng=None, k_points=None):
        rng = np.random.default_rng(1) if rng is None else rng
        self.eigenvalues = np.sort(rng.normal(size=states))
        self.occupations = np.r_[np.ones(occupied), np.zeros(states - occupied)]
        vectors = rng.normal(size=(grid.size, states))
        self.wavefunctions = vectors / np.linalg.norm(vectors, axis=0)
        self.next_effective_potential = rng.normal(size=grid.size)
        self.density = np.abs(rng.normal(size=grid.size))
        self.k_points = k_points
        self.k_weights = None if k_points is None else np.full(len(k_points), 1.0 / len(k_points))
        if k_points is not None:
            count = len(k_points)
            self.eigenvalues = np.tile(self.eigenvalues, count)
            self.occupations = np.tile(self.occupations, count)


class RecordLayoutTests(unittest.TestCase):
    def test_cluster_file_has_the_records_of_wfnsave(self) -> None:
        grid = _cluster_grid()
        result = _Result(grid)
        data, notes = dat.parsec_dat_from_result(
            result, grid, WavefunctionSelection(), date_label="27-OCT-2026 14:01:07 -0700"
        )
        self.assertEqual(notes, ())
        with tempfile.TemporaryDirectory() as directory:
            path = dat.write_parsec_dat(Path(directory) / "parsec.dat", data)
            with FortranFile(path, "r", header_dtype=np.uint32) as stream:
                self.assertEqual(stream.read_record("S26")[0], b"27-OCT-2026 14:01:07 -0700")
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_allclose(stream.read_reals(np.float64), [0.8, 3.0])
                self.assertEqual(stream.read_ints(np.int32)[0], grid.size)
                np.testing.assert_allclose(stream.read_reals(np.float64), [0.5, 0.5, 0.5])
                np.testing.assert_array_equal(stream.read_ints(np.int32), [grid.size, 1])
                np.testing.assert_array_equal(stream.read_ints(np.int32), np.eye(3).ravel())
                np.testing.assert_allclose(stream.read_reals(np.float64), np.eye(3).ravel())
                np.testing.assert_allclose(stream.read_reals(np.float64), [0, 0, 0])
                lattice = stream.read_reals(np.float64)
                np.testing.assert_allclose(lattice[:9], (12.0 * np.eye(3)).ravel())
                np.testing.assert_allclose(lattice[9:], (np.eye(3) / 12.0).ravel())
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1])
                points = stream.read_ints(np.int32).reshape(-1, 3)
                np.testing.assert_array_equal(points, grid.integer_coordinates)
                self.assertEqual(stream.read_ints(np.int32)[0], 4)
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 1, 1, 1])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.eigenvalues)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.occupations)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.next_effective_potential)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.density)
                for _ in range(7):
                    stream.read_record(np.uint8)
                self.assertEqual(stream.read_ints(np.int32)[0], 2)
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 2])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.wavefunctions[:, 0])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.wavefunctions[:, 1])
                with self.assertRaises(Exception):
                    stream.read_record(np.uint8)

    def test_round_trip_of_spin_channels_and_k_points(self) -> None:
        grid = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.8])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        rng = np.random.default_rng(3)
        channels = tuple(
            SpinChannelResult(
                eigenvalues=np.sort(rng.normal(size=3)),
                occupations=np.array([1.0, 0.5, 0.0]),
                wavefunctions=np.linalg.qr(rng.normal(size=(grid.size, 3)))[0],
                representations=np.ones(3, dtype=np.int32),
                density=np.abs(rng.normal(size=grid.size)),
                xc_potential=np.zeros(grid.size),
                input_effective_potential=np.zeros(grid.size),
                output_effective_potential=np.zeros(grid.size),
                next_effective_potential=rng.normal(size=grid.size),
                electron_count=1.5,
            )
            for _ in range(2)
        )

        class Spin:
            spin = channels
            k_points = None

        data, _ = dat.parsec_dat_from_result(Spin(), grid, WavefunctionSelection(mode="all"))
        with tempfile.TemporaryDirectory() as directory:
            path = dat.write_parsec_dat(Path(directory) / "parsec.dat", data)
            back = dat.read_parsec_dat(path)
        self.assertEqual(back.spin_count, 2)
        self.assertEqual(back.boundary_code, 1)
        # a cell without k-points: kpnum 1, k = 0, weight 1 and PARSEC's kptmethod -1 (0 under
        # PARSEC_GAMMA_PARSEC_DAT=off, as before)
        self.assertEqual(back.periodic.kpoint_method, dat.KPOINT_NONE)
        np.testing.assert_array_equal(back.periodic.k_points, np.zeros((1, 3)))
        np.testing.assert_array_equal(back.periodic.k_weights, np.ones(1))
        with patch.dict("os.environ", {dat.GAMMA_PARSEC_DAT_SWITCH: "off"}):
            data_off, _ = dat.parsec_dat_from_result(Spin(), grid, WavefunctionSelection(mode="all"))
        self.assertEqual(data_off.periodic.kpoint_method, 0)
        np.testing.assert_allclose(back.periodic.lattice_vectors, np.diag([4.0, 4.0, 4.8]))
        np.testing.assert_allclose(back.step, [0.8, 0.8, 0.8])
        self.assertEqual(back.rmax, 1.0)
        for spin in range(2):
            np.testing.assert_array_equal(back.potentials[spin], channels[spin].next_effective_potential)
            np.testing.assert_array_equal(back.densities[spin], channels[spin].density)
            np.testing.assert_array_equal(back.wavefunctions[spin][0].state_numbers, [1, 2, 3])
            np.testing.assert_allclose(back.wavefunctions[spin][0].vectors, channels[spin].wavefunctions)
        # PARSEC's periodic integers run from -n/2 to n-n/2-1 with the same
        # positions as the Python integers 0..n-1, modulo the cell.
        counts = grid.points_per_axis
        self.assertTrue(np.all(back.points >= -(counts // 2)))
        self.assertTrue(np.all(back.points < counts - counts // 2))
        np.testing.assert_array_equal(back.points % counts, grid.integer_coordinates)

        k_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _Result(grid, k_points=k_points)
        for setting, phrase in (
            ("off", "k-point drivers keep no orbitals"),
            ("auto", "its k-point loop kept no orbitals"),
        ):
            with self.subTest(setting=setting), patch.dict("os.environ", {dat.KPOINT_WAVEFUNCTIONS_SWITCH: setting}):
                data, notes = dat.parsec_dat_from_result(
                    result, grid, WavefunctionSelection(), monkhorst_pack_grid=(2, 1, 1)
                )
                self.assertTrue(any(phrase in note for note in notes))
                with tempfile.TemporaryDirectory() as directory:
                    back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
                self.assertEqual(back.wavefunction_kind, 1)
                self.assertEqual(back.periodic.kpoint_method, dat.KPOINT_MONKHORST_PACK)
                np.testing.assert_allclose(back.periodic.k_points, k_points)
                np.testing.assert_allclose(back.periodic.k_weights, [0.5, 0.5])
                self.assertEqual(len(back.levels[0]), 2)
                self.assertEqual([block.state_numbers.size for block in back.wavefunctions[0]], [0, 0])

    def test_kpoint_loops_write_the_kept_orbitals_of_every_kpoint(self) -> None:
        # PARSEC_KPOINT_WAVEFUNCTIONS=auto: wfnsave.F90 writes the selected states of every (spin, k-point) block,
        # each k-point's columns ordered by its own levels; off keeps the file of before (no orbitals).
        grid = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.8])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        rng = np.random.default_rng(11)
        states = 4
        k_points = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.1]])

        def orbitals():
            vectors = rng.normal(size=(grid.size, states)) + 1j * rng.normal(size=(grid.size, states))
            return np.linalg.qr(vectors)[0]

        def levels():
            # unsorted within each k-point: the file sorts them, and its columns follow
            return np.concatenate([rng.permutation(np.sort(rng.normal(size=states))) for _ in k_points])

        occupied = np.tile([1.0, 1.0, 0.0, 0.0], len(k_points))

        class Scalar:
            eigenvalues = levels()
            occupations = occupied.copy()
            next_effective_potential = rng.normal(size=grid.size)
            density = np.abs(rng.normal(size=grid.size))
            k_weights = np.full(3, 1.0 / 3.0)
            k_wavefunctions = tuple(orbitals() for _ in range(3))

        Scalar.k_points = k_points

        class SpinK:
            eigenvalues_up = levels()
            eigenvalues_down = levels()
            occupations_up = occupied.copy()
            occupations_down = np.tile([1.0, 0.0, 0.0, 0.0], 3)
            density_up = np.abs(rng.normal(size=grid.size))
            density_down = np.abs(rng.normal(size=grid.size))
            scf_potentials = {"potential_up": rng.normal(size=grid.size), "potential_down": rng.normal(size=grid.size)}
            next_potentials = None
            k_weights = np.full(3, 1.0 / 3.0)
            k_wavefunctions = (tuple(orbitals() for _ in range(3)), tuple(orbitals() for _ in range(3)))

        SpinK.k_points = k_points
        cases = (
            (Scalar(), (Scalar.eigenvalues,), (Scalar.occupations,), (Scalar.k_wavefunctions,)),
            (
                SpinK(),
                (SpinK.eigenvalues_up, SpinK.eigenvalues_down),
                (SpinK.occupations_up, SpinK.occupations_down),
                SpinK.k_wavefunctions,
            ),
        )
        for result, values, occupations, kept in cases:
            for selection, expected in ((WavefunctionSelection(mode="all"), None), (WavefunctionSelection(), "occ")):
                with self.subTest(result=type(result).__name__, selection=selection.mode):
                    data, notes = dat.parsec_dat_from_result(result, grid, selection)
                    self.assertFalse(any("not in parsec.dat" in note for note in notes))
                    with tempfile.TemporaryDirectory() as directory:
                        back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
                    self.assertEqual(back.wavefunction_kind, 1)
                    for spin in range(len(values)):
                        orders = [
                            np.argsort(values[spin][k * states : (k + 1) * states], kind="stable") for k in range(3)
                        ]
                        count = states if expected is None else dat.highest_occupied(
                            [occupations[spin][k * states : (k + 1) * states][orders[k]] for k in range(3)]
                        )
                        for k in range(3):
                            block = back.wavefunctions[spin][k]
                            np.testing.assert_array_equal(block.state_numbers, np.arange(1, count + 1))
                            np.testing.assert_allclose(block.vectors, kept[spin][k][:, orders[k][:count]])
            with self.subTest(result=type(result).__name__, setting="off"), patch.dict(
                "os.environ", {dat.KPOINT_WAVEFUNCTIONS_SWITCH: "off"}
            ):
                data, notes = dat.parsec_dat_from_result(result, grid, WavefunctionSelection(mode="all"))
                self.assertTrue(any("k-point drivers keep no orbitals" in note for note in notes))
                self.assertEqual(
                    [block.state_numbers.size for channel in data.wavefunctions for block in channel],
                    [0] * (3 * len(values)),
                )
        # a block without orbitals of the whole grid (a sector or row block): none written, with the note
        partial = SpinK()
        partial.k_wavefunctions = (SpinK.k_wavefunctions[0], (None,) * 3)
        data, notes = dat.parsec_dat_from_result(partial, grid, WavefunctionSelection(mode="all"))
        self.assertEqual([block.state_numbers.size for block in data.wavefunctions[0]], [states] * 3)
        self.assertEqual([block.state_numbers.size for block in data.wavefunctions[1]], [0] * 3)
        self.assertTrue(any("the MPI-distributed solvers gather none" in note for note in notes))
        # the lowest columns of each k-point (a Save_Wfn_Bands list keeps those up to its highest state): written in
        # place; a state beyond them is not in the file, with the note
        lowest = Scalar()
        lowest.eigenvalues = np.concatenate([np.sort(rng.normal(size=states)) for _ in k_points])
        lowest.k_wavefunctions = tuple(block[:, :3] for block in Scalar.k_wavefunctions)
        data, notes = dat.parsec_dat_from_result(lowest, grid, WavefunctionSelection(mode="list", indices=(1, 3)))
        self.assertIsNotNone(data.wavefunctions[0][0].columns)
        with tempfile.TemporaryDirectory() as directory:
            back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
        for k in range(3):
            np.testing.assert_array_equal(back.wavefunctions[0][k].state_numbers, [1, 3])
            np.testing.assert_allclose(back.wavefunctions[0][k].vectors, Scalar.k_wavefunctions[k][:, [0, 2]])
            np.testing.assert_allclose(data.wavefunctions[0][k].selected_vectors, Scalar.k_wavefunctions[k][:, [0, 2]])
        data, notes = dat.parsec_dat_from_result(lowest, grid, WavefunctionSelection(mode="list", indices=(4,)))
        self.assertTrue(any("do not hold the selected states" in note for note in notes))
        self.assertEqual([block.state_numbers.size for block in data.wavefunctions[0]], [0, 0, 0])

    def test_a_gamma_point_input_on_the_kpoint_loop_writes_real_wavefunctions_of_kind_zero(self) -> None:
        # PARSEC_GAMMA_PARSEC_DAT=auto: an input without k-points (Fortran's nkpt = 0, real wavefunctions) that runs
        # as the 1x1x1 k-point mesh writes the file of the Gamma point; off: the complex file of a k-point run.
        grid = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.8])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        rng = np.random.default_rng(13)
        states = 6
        # a real symmetric Hamiltonian with a triplet and a pair, and complex eigenvectors of it: random unitary
        # mixtures inside each multiplet and random phases, as a complex solver returns them
        levels = np.array([-1.0, -0.4, -0.4, -0.4, 0.3, 0.3])
        basis = np.linalg.qr(rng.normal(size=(grid.size, states)))[0]
        mixture = np.zeros((states, states), dtype=complex)
        for block in ([0], [1, 2, 3], [4, 5]):
            size = len(block)
            unitary = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))[0]
            mixture[np.ix_(block, block)] = unitary
        complex_orbitals = basis @ mixture
        # (the solver's levels of a multiplet differ by round-off; in ascending order)
        values = levels + 1.0e-9 * np.array([0.0, 0.0, 1.0, 2.0, 0.0, 1.0])

        class SpinGamma:
            eigenvalues_up = values.copy()
            eigenvalues_down = values.copy()
            occupations_up = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
            occupations_down = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            density_up = np.abs(rng.normal(size=grid.size))
            density_down = np.abs(rng.normal(size=grid.size))
            scf_potentials = {"potential_up": rng.normal(size=grid.size), "potential_down": rng.normal(size=grid.size)}
            next_potentials = (rng.normal(size=grid.size), rng.normal(size=grid.size))
            k_points = np.zeros((1, 3))
            k_weights = np.ones(1)
            k_wavefunctions = ((complex_orbitals,), (complex_orbitals,))

        hamiltonian_columns = basis * levels[None, :]
        for setting in ("auto", "off"):
            with self.subTest(setting=setting), patch.dict("os.environ", {dat.GAMMA_PARSEC_DAT_SWITCH: setting}):
                data, notes = dat.parsec_dat_from_result(
                    SpinGamma(), grid, WavefunctionSelection(mode="all"), gamma_point=True
                )
                self.assertFalse(any("span the complex orbitals" in note for note in notes))
                with tempfile.TemporaryDirectory() as directory:
                    back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
                self.assertEqual(back.wavefunction_kind, 0 if setting == "auto" else 1)
                self.assertEqual(
                    back.periodic.kpoint_method, dat.KPOINT_NONE if setting == "auto" else dat.KPOINT_MANUAL
                )
                np.testing.assert_array_equal(back.periodic.k_points, np.zeros((1, 3)))
                for spin in range(2):
                    vectors = back.wavefunctions[spin][0].vectors
                    np.testing.assert_array_equal(back.wavefunctions[spin][0].state_numbers, np.arange(1, 7))
                    if setting == "off":
                        np.testing.assert_allclose(vectors, complex_orbitals)
                        continue
                    self.assertEqual(vectors.dtype, np.float64)
                    np.testing.assert_allclose(vectors.T @ vectors, np.eye(states), atol=1e-12)
                    # eigenvectors of the real Hamiltonian H = basis diag(levels) basis^T, level by level
                    applied = hamiltonian_columns @ (basis.T @ vectors)
                    np.testing.assert_allclose(applied, vectors * np.sort(levels)[None, :], atol=1e-8)
        # the occupied states only (default selection), and no conversion where the input has k-points
        data, _ = dat.parsec_dat_from_result(SpinGamma(), grid, WavefunctionSelection(), gamma_point=True)
        self.assertEqual([channel[0].state_numbers.size for channel in data.wavefunctions], [4, 2])
        self.assertEqual(data.wavefunction_kind, 0)
        data, _ = dat.parsec_dat_from_result(SpinGamma(), grid, WavefunctionSelection(), gamma_point=False)
        self.assertEqual(data.wavefunction_kind, 1)
        # a multiplet cut by the number of states: the real orbitals leave a part outside, which a note gives
        cut = SpinGamma()
        cut.eigenvalues_up = cut.eigenvalues_down = values[:5].copy()
        cut.occupations_up = cut.occupations_down = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        cut.k_wavefunctions = ((complex_orbitals[:, :5],), (complex_orbitals[:, :5],))
        data, notes = dat.parsec_dat_from_result(cut, grid, WavefunctionSelection(mode="all"), gamma_point=True)
        self.assertTrue(any("span the complex orbitals" in note for note in notes))
        # the cut multiplet among empty states that the file does not write (the occupied states only): no note,
        # and the written real orbitals span the complex ones of their states
        data, notes = dat.parsec_dat_from_result(cut, grid, WavefunctionSelection(), gamma_point=True)
        self.assertEqual([channel[0].state_numbers.size for channel in data.wavefunctions], [4, 4])
        self.assertFalse(any("span the complex orbitals" in note for note in notes))
        written = data.wavefunctions[0][0].vectors
        captured = np.linalg.norm(written.T @ complex_orbitals[:, :4], axis=0)
        np.testing.assert_allclose(captured, np.ones(4), atol=1e-12)

    def test_real_gamma_orbitals_and_the_gamma_point_input(self) -> None:
        rng = np.random.default_rng(17)
        real = np.linalg.qr(rng.normal(size=(50, 3)))[0]
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=3))
        made, outside = dat.real_gamma_orbitals(real * phases[None, :], np.array([-1.0, 0.0, 1.0]))
        self.assertLess(outside, 1e-12)
        np.testing.assert_allclose(np.abs(np.sum(made * real, axis=0)), np.ones(3), atol=1e-12)
        # a real input is returned as it is
        same, outside = dat.real_gamma_orbitals(real, np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(same, real)
        self.assertEqual(outside, 0.0)
        # a doublet cut to one of its complex partners: its part outside counts only where its column is named
        pair = np.linalg.qr(np.concatenate([real, rng.normal(size=(50, 1))], axis=1))[0][:, 3]
        cut = np.concatenate(
            [real[:, :2] * phases[None, :2], (real[:, 2:3] + 1j * pair[:, None]) / np.sqrt(2.0)], axis=1
        )
        _, every = dat.real_gamma_orbitals(cut, np.array([-1.0, 0.0, 1.0]))
        _, written = dat.real_gamma_orbitals(cut, np.array([-1.0, 0.0, 1.0]), columns=[0, 1])
        self.assertAlmostEqual(every, np.sqrt(0.5), places=12)
        self.assertLess(written, 1e-12)
        cluster = PeriodicCell(np.diag([4.0, 4.0, 4.0]))

        class Problem:
            periodic_cell = cluster
            has_kpoint_sampling = False

        self.assertTrue(dat.gamma_point_input(Problem()))
        Problem.has_kpoint_sampling = True
        self.assertFalse(dat.gamma_point_input(Problem()))
        Problem.periodic_cell = None
        self.assertFalse(dat.gamma_point_input(Problem()))
        self.assertFalse(dat.gamma_point_input(None))
        for name in (dat.KPOINT_WAVEFUNCTIONS_SWITCH, dat.GAMMA_PARSEC_DAT_SWITCH):
            with self.subTest(name=name), patch.dict("os.environ", {name: "on"}):
                with self.assertRaisesRegex(ValueError, f"{name} must be auto or off"):
                    dat.validate_settings()

    def test_spin_kpoint_loop_of_one_kpoint_writes_its_kept_orbitals_and_mixed_potentials(self) -> None:
        # The spin k-point loop with keep_wavefunctions (the ionic loops of a spin cell at the Gamma point,
        # PARSEC_SPIN_IONIC_KPOINT): its Bloch orbitals of k = 0 and the mixed potentials (vnew) go to parsec.dat.
        grid = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.8])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        rng = np.random.default_rng(5)
        states = 4

        def orbitals():
            vectors = rng.normal(size=(grid.size, states)) + 1j * rng.normal(size=(grid.size, states))
            return np.linalg.qr(vectors)[0]

        class SpinK:
            eigenvalues_up = np.sort(rng.normal(size=states))
            eigenvalues_down = np.sort(rng.normal(size=states))
            occupations_up = np.array([1.0, 1.0, 0.0, 0.0])
            occupations_down = np.array([1.0, 0.0, 0.0, 0.0])
            density_up = np.abs(rng.normal(size=grid.size))
            density_down = np.abs(rng.normal(size=grid.size))
            scf_potentials = {"potential_up": rng.normal(size=grid.size), "potential_down": rng.normal(size=grid.size)}
            next_potentials = (rng.normal(size=grid.size), rng.normal(size=grid.size))
            k_points = np.zeros((1, 3))
            k_weights = np.ones(1)
            k_wavefunctions = ((orbitals(),), (orbitals(),))

        result = SpinK()
        data, notes = dat.parsec_dat_from_result(result, grid, WavefunctionSelection(mode="all"))
        self.assertEqual(notes, ())
        with tempfile.TemporaryDirectory() as directory:
            back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
        self.assertEqual(back.wavefunction_kind, 1)
        for spin in range(2):
            np.testing.assert_array_equal(back.potentials[spin], result.next_potentials[spin])
            np.testing.assert_array_equal(back.wavefunctions[spin][0].state_numbers, [1, 2, 3, 4])
            np.testing.assert_allclose(back.wavefunctions[spin][0].vectors, result.k_wavefunctions[spin][0])
        # the default selection: the occupied states of each channel
        data, _ = dat.parsec_dat_from_result(result, grid, WavefunctionSelection())
        self.assertEqual([channel[0].state_numbers.size for channel in data.wavefunctions], [2, 1])
        # without kept orbitals or mixed potentials (a single point of the loop), as before; under
        # PARSEC_KPOINT_WAVEFUNCTIONS=off several k-points keep no orbitals (one set of columns was written for
        # every k-point), under auto each k-point writes its own
        for setting, changes in (
            ("auto", {"k_wavefunctions": None, "next_potentials": None}),
            ("off", {"k_wavefunctions": None, "next_potentials": None}),
            ("off", {"k_points": np.zeros((2, 3)), "k_weights": np.full(2, 0.5),
                     "eigenvalues_up": np.tile(SpinK.eigenvalues_up, 2),
                     "eigenvalues_down": np.tile(SpinK.eigenvalues_down, 2),
                     "occupations_up": np.tile(SpinK.occupations_up, 2),
                     "occupations_down": np.tile(SpinK.occupations_down, 2),
                     "k_wavefunctions": ((orbitals(), orbitals()), (orbitals(), orbitals()))}),
            ("auto", {"k_wavefunctions": ((orbitals()[: grid.size - 1],), (orbitals()[: grid.size - 1],))}),
            ("off", {"k_wavefunctions": ((orbitals()[: grid.size - 1],), (orbitals()[: grid.size - 1],))}),
        ):
            with self.subTest(setting=setting, changes=sorted(changes)), patch.dict(
                "os.environ", {dat.KPOINT_WAVEFUNCTIONS_SWITCH: setting}
            ):
                other = SpinK()
                for name, value in changes.items():
                    setattr(other, name, value)
                data, notes = dat.parsec_dat_from_result(other, grid, WavefunctionSelection(mode="all"))
                phrase = (
                    "k-point drivers keep no orbitals" if setting == "off"
                    else "its k-point loop kept no orbitals" if other.k_wavefunctions is None
                    else "not on the whole grid"
                )
                self.assertTrue(any(phrase in note for note in notes))
                self.assertEqual([block.state_numbers.size for channel in data.wavefunctions for block in channel],
                                 [0] * (2 * len(other.k_points)))
                expected = (
                    other.next_potentials
                    if other.next_potentials is not None
                    else (other.scf_potentials["potential_up"], other.scf_potentials["potential_down"])
                )
                for spin in range(2):
                    np.testing.assert_array_equal(data.potentials[spin], expected[spin])

    def test_long_records_are_split_into_gfortran_subrecords(self) -> None:
        values = np.arange(40, dtype="<f8")
        stream = io.BytesIO()
        with patch.object(dat, "_MAXIMUM_SUBRECORD", 96):
            dat.write_record(stream, values)
        raw = stream.getvalue()
        markers = []
        offset = 0
        while offset < len(raw):
            head = int(np.frombuffer(raw[offset : offset + 4], "<i4")[0])
            size = abs(head)
            tail = int(np.frombuffer(raw[offset + 4 + size : offset + 8 + size], "<i4")[0])
            markers.append((head, tail))
            offset += size + 8
        self.assertEqual(markers, [(-96, 96), (-96, -96), (-96, -96), (32, -32)])
        stream.seek(0)
        np.testing.assert_array_equal(np.frombuffer(dat.read_record(stream), "<f8"), values)

    def test_occupied_states_follow_ifmax_of_the_last_k_point(self) -> None:
        self.assertEqual(dat.highest_occupied([np.array([1.0, 1e-14, 0.0])]), 2)
        self.assertEqual(dat.highest_occupied([np.array([1, 1, 1.0]), np.array([1.0, 0, 0])]), 1)
        self.assertEqual(dat.highest_occupied([np.zeros(3)]), 0)
        with self.assertRaisesRegex(ValueError, "Save_Wfn_Bands asks for state 5"):
            dat.selected_states(WavefunctionSelection(mode="list", indices=(5,)), [np.ones(4)], 4)


def _symmetric_field(positions):
    x, y, z = positions.T
    return np.exp(-((x - 0.0) ** 2) - 2.0 * (y - 0.3) ** 2 - 0.5 * z**2) + 0.1 * x**2 * z**2


def _fortran_wedge(grid, symmetry):
    """The irreducible wedge as grid_partition.f90 builds it."""

    steps = dat.grid_steps(grid)
    shift = np.asarray(grid.settings.shift)
    wedge = {}
    order = []
    for point in grid.integer_coordinates:
        position = (point + shift) * steps
        found = None
        for operation in range(symmetry.ntrans):
            image = symmetry.apply(operation, position[None, :])[0]
            integers = tuple(np.rint(image / steps - shift).astype(int))
            if integers in wedge:
                found = integers
                break
        if found is None:
            wedge[tuple(point)] = len(order)
            order.append(tuple(point))
    return np.asarray(order, dtype=np.int64)


class GridMappingTests(unittest.TestCase):
    def test_a_symmetry_reduced_file_unfolds_onto_the_full_grid(self) -> None:
        grid = _cluster_grid(spacing=0.6, radius=3.0)
        alatt = dat.cluster_alatt(3.0)
        mirrors = [np.diag(d) for d in ((1, 1, 1), (-1, 1, -1), (-1, 1, 1), (1, 1, -1))]
        symmetry = dat.ParsecDatSymmetry(
            rmtrx=np.asarray(mirrors, dtype=np.int32),
            trans=np.asarray(mirrors, dtype=float),
            tnp=np.zeros((4, 3)),
            alatt=alatt,
            invlat=np.linalg.inv(alatt.T),
            chi=np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]),
        )
        wedge = _fortran_wedge(grid, symmetry)
        self.assertLess(len(wedge), grid.size / 3)
        positions = (wedge + 0.5) * 0.6
        values = _symmetric_field(positions)
        data = dat.ParsecDat(
            date_label=dat.fortran_date_label(),
            system_type=dat.system_type(1, 0, 0),
            step=np.full(3, 0.6),
            rmax=3.0,
            ndim=grid.size,
            shift=np.full(3, 0.5),
            symmetry=symmetry,
            points=wedge,
            levels=((dat.ParsecDatLevels(np.ones(1, np.int32), np.zeros(1), np.ones(1)),),),
            potentials=(values,),
            densities=(2.0 * values,),
        )
        with tempfile.TemporaryDirectory() as directory:
            back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
        np.testing.assert_array_equal(back.symmetry.chi, symmetry.chi)
        mapping = dat.grid_rows_of_file(back, grid)
        self.assertEqual(mapping.missing, 0)
        np.testing.assert_allclose(mapping.field(back.potentials[0]), _symmetric_field(grid.coordinates), atol=1e-12)

    def test_restart_checks_and_a_smaller_stored_sphere(self) -> None:
        small = _cluster_grid(radius=2.4)
        large = _cluster_grid(radius=3.0)
        result = _Result(small)
        data, _ = dat.parsec_dat_from_result(result, small, WavefunctionSelection())
        fields = restart_fields(data, small, spin_count=1)
        self.assertFalse(fields.extrapolate)
        np.testing.assert_array_equal(fields.potentials[0], result.next_effective_potential)
        grown = restart_fields(data, large, spin_count=1)
        self.assertTrue(grown.extrapolate)
        self.assertEqual(grown.missing_points, large.size - small.size)
        rows = large.rows_for_integer_coordinates(small.integer_coordinates)
        np.testing.assert_array_equal(grown.densities[0][rows], result.density)
        self.assertEqual(float(np.sum(np.delete(grown.densities[0], rows))), 0.0)
        shrunk = restart_fields(
            dat.parsec_dat_from_result(_Result(large), large, WavefunctionSelection())[0],
            small,
            spin_count=1,
        )
        self.assertEqual(shrunk.dropped_points, large.size - small.size)
        for kwargs, code in (
            ({"spin_count": 2}, 361),
            ({"spin_count": 1, "complex_wavefunctions": True}, 362),
        ):
            with self.assertRaises(RestartError) as caught:
                restart_fields(data, small, **kwargs)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(RestartError) as caught:
            restart_fields(data, _cluster_grid(spacing=0.7, radius=2.4), spin_count=1)
        self.assertEqual(caught.exception.code, 366)
        with self.assertRaisesRegex(RestartError, "Ignore_Symmetry"):
            restart_fields(data, _cluster_grid(radius=2.4, shift=(0.0, 0.0, 0.0)), spin_count=1)
        periodic = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.0])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        with self.assertRaises(RestartError) as caught:
            restart_fields(data, periodic, spin_count=1)
        self.assertEqual(caught.exception.code, 363)
        # a cell restarted on another grid (another rule): error 364 names both counts and the spacing of this run
        coarse = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.0])), PeriodicGridSettings(spacing=0.5, expansion_order=4)
        )
        stored, _ = dat.parsec_dat_from_result(_Result(periodic), periodic, WavefunctionSelection())
        with self.assertRaises(RestartError) as caught:
            restart_fields(stored, coarse, spin_count=1)
        self.assertEqual(caught.exception.code, 364)
        message = str(caught.exception)
        self.assertIn(f"{[int(v) for v in periodic.points_per_axis]} points along the lattice vectors", message)
        self.assertIn(f"this run {[int(v) for v in coarse.points_per_axis]}", message)
        self.assertIn("Grid_Spacing 0.500000 bohr", message)
        self.assertIn("default routines", message)
        # steps that differ between the lattice vectors: no one spacing gives them, the routine set does
        from parsec_python.SCF.restart import _grid_difference

        text = _grid_difference(stored, coarse, np.array([0.5, 0.5, 0.6]))
        self.assertIn("Use_Fortran_Routines: .false.", text)
        self.assertNotIn("Grid_Spacing", text)
        # a run of the default routines names the switch that reads a file of the Fortran-derived routines, and says
        # that the file is one of theirs where its counts are those these routines build for the input
        self.assertIn("restarts with PARSEC_FILE_KEYWORD_ROUTINES=fortran or Use_Fortran_Routines: .true.", message)
        stored_counts = [int(v) for v in periodic.points_per_axis]
        with self.assertRaises(RestartError) as caught:
            restart_fields(stored, coarse, spin_count=1, fortran_counts=lambda: stored_counts)
        self.assertIn("it is one of theirs", str(caught.exception))
        self.assertIn("restart it with PARSEC_FILE_KEYWORD_ROUTINES=fortran or Use_Fortran_Routines: .true.",
                      str(caught.exception))
        for hint in (lambda: [1, 2, 3], lambda: None):
            with self.assertRaises(RestartError) as caught:
                restart_fields(stored, coarse, spin_count=1, fortran_counts=hint)
            self.assertNotIn("one of theirs", str(caught.exception))
        # a run on the Fortran-derived routines: the message of before
        fortran = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.0])),
            PeriodicGridSettings(spacing=0.5, expansion_order=4, fortran_routines=True),
        )
        with self.assertRaises(RestartError) as caught:
            restart_fields(stored, fortran, spin_count=1, fortran_counts=lambda: stored_counts)
        self.assertNotIn("PARSEC_FILE_KEYWORD_ROUTINES", str(caught.exception))
        # the same grid restarts as before
        self.assertFalse(check_restart_file(stored, periodic, spin_count=1, complex_wavefunctions=False))

    def test_a_file_that_decides_the_points_on_the_sphere_otherwise_restarts(self) -> None:
        # Grid_Shift 0, R = 21 h: points lie exactly on the sphere, and PARSEC_SPHERE_TIES decides them by its rule
        import os

        from parsec_python.Output.wfn_dat import wfn_dat_from_result
        from parsec_python.SCF.restart import _wfn_dat_tie_difference, check_wfn_dat, restart_report_lines

        grids = {}
        for rule in ("dot", "fortran", "symmetric"):
            with patch.dict(os.environ, {"PARSEC_SPHERE_TIES": rule}):
                grids[rule] = _cluster_grid(spacing=0.4, radius=8.4, shift=(0.0, 0.0, 0.0))
        self.assertEqual((grids["dot"].size, grids["symmetric"].size), (38807, 38839))
        current = grids["symmetric"]
        for rule in ("dot", "fortran"):
            stored = grids[rule]
            result = _Result(stored)
            data, _ = dat.parsec_dat_from_result(result, stored, WavefunctionSelection())
            mine = {tuple(p) for p in current.integer_coordinates.tolist()}
            theirs = {tuple(p) for p in stored.integer_coordinates.tolist()}
            expected = (len(mine - theirs), len(theirs - mine))
            self.assertGreater(sum(expected), 0)
            with self.subTest(file=rule):
                self.assertFalse(check_restart_file(data, current, spin_count=1, complex_wavefunctions=False))
                fields = restart_fields(data, current, spin_count=1)
                self.assertEqual(fields.sphere_ties, expected)
                rows = current.rows_for_integer_coordinates(stored.integer_coordinates)
                kept = rows >= 0
                np.testing.assert_array_equal(fields.potentials[0][rows[kept]], result.next_effective_potential[kept])
                held = np.zeros(current.size, dtype=bool)
                held[rows[kept]] = True
                self.assertEqual(float(np.abs(fields.potentials[0][~held]).sum()), 0.0)
                self.assertTrue(any("PARSEC_SPHERE_TIES=symmetric" in line for line in restart_report_lines(fields)))
                # wfn.dat (restart.F90): the same
                wfn = wfn_dat_from_result(result, stored)
                self.assertFalse(check_wfn_dat(wfn, current))
                self.assertEqual(_wfn_dat_tie_difference(wfn, current), expected)
        # a file of the same grid: nothing to note
        same, _ = dat.parsec_dat_from_result(_Result(current), current, WavefunctionSelection())
        self.assertIsNone(restart_fields(same, current, spin_count=1).sphere_ties)
        # a point off the sphere is another grid: the errors of before
        stored = grids["dot"]
        inner = int(np.argmin(np.einsum("ij,ij->i", stored.coordinates, stored.coordinates)))
        data, _ = dat.parsec_dat_from_result(_Result(stored), stored, WavefunctionSelection())
        keep = np.ones(stored.size, dtype=bool)
        keep[inner] = False
        from dataclasses import replace

        holed = replace(
            data, ndim=stored.size - 1, points=np.asarray(data.points)[keep],
            potentials=tuple(np.asarray(v)[keep] for v in data.potentials),
            densities=tuple(np.asarray(v)[keep] for v in data.densities),
        )
        with self.assertRaises(RestartError) as caught:
            restart_fields(holed, current, spin_count=1)
        self.assertEqual(caught.exception.code, 367)
        wfn = wfn_dat_from_result(_Result(stored), stored)
        holed_wfn = replace(wfn, ndim=stored.size - 1, points=np.asarray(wfn.points)[keep])
        with self.assertRaises(RestartError) as caught:
            check_wfn_dat(holed_wfn, current)
        self.assertEqual(caught.exception.code, 326)
        # the same radius on a grid shifted by half a step has no point on the sphere: another grid
        with self.assertRaisesRegex(RestartError, "Ignore_Symmetry"):
            restart_fields(data, _cluster_grid(spacing=0.4, radius=8.4), spin_count=1)

    def test_warm_start_takes_the_stored_potential_or_rebuilds_it(self) -> None:
        grid = _cluster_grid()
        result = _Result(grid)
        data, _ = dat.parsec_dat_from_result(result, grid, WavefunctionSelection())
        fields = restart_fields(data, grid, spin_count=1)

        class System:
            ionic_potential = -np.ones(grid.size)

        start = restart_warm_start(System(), fields)
        np.testing.assert_array_equal(start.input_potential, result.next_effective_potential)
        np.testing.assert_array_equal(start.hartree_potential, np.ones(grid.size))
        self.assertIsNone(start.eigensolver_state)
        self.assertIsNone(start.density_guess)
        rebuilt = restart_warm_start(System(), type(fields)(**{**fields.__dict__, "extrapolate": True}))
        np.testing.assert_array_equal(rebuilt.input_potential, System.ionic_potential)
        np.testing.assert_array_equal(rebuilt.density_guess, result.density)
        np.testing.assert_array_equal(rebuilt.hxc_reference, np.zeros(grid.size))

    def test_only_the_first_loop_of_an_ionic_run_takes_the_restart(self) -> None:
        from parsec_python.Relaxation.backend import IonicBackend

        seen = []
        backend = IonicBackend(
            run_scf=lambda system, callback=None, warm_start=None: seen.append(warm_start),
            compute_forces=None,
            update_geometry=None,
            warm_start=None,
        )
        restarted = with_first_warm_start(backend, "restart")
        restarted.run_scf("system", callback=None, warm_start=None)
        restarted.run_scf("system", callback=None, warm_start=None)
        restarted.run_scf("system", callback=None, warm_start="moved")
        self.assertEqual(seen, ["restart", None, "moved"])


class EigenDatTests(unittest.TestCase):
    # Lines of eigen.dat that PARSEC wrote for H2O (Ignore_Symmetry).
    FORTRAN = [
        "  State     Eigenvalue [Ry]    Eigenvalue [eV]    Occupation   Repr.   Movement   spin",
        "    1       -2.2076030485      -30.0362        1.0000   1      0    ",
        "    5       -0.5433746657       -7.3930        0.0000   1      0    ",
    ]

    def test_header_and_rows_are_those_of_eigensave(self) -> None:
        self.assertEqual(eigen_dat_header(False), self.FORTRAN[0])
        self.assertEqual(
            eigen_dat_header(True),
            "  State     Eigenvalue [Ry]    Eigenvalue [eV]    Occupation   Repr.   k-point  Movement   spin",
        )
        self.assertEqual(eigen_dat_line(1, -2.2076030485, 1.0, 1, 0), self.FORTRAN[1])
        self.assertEqual(eigen_dat_line(5, -0.5433746657, 8.79e-13, 1, 0), self.FORTRAN[2])
        self.assertEqual(
            eigen_dat_line(3, -0.25, 0.5, 2, 4, "dn", kpoint=7),
            "    3       -0.2500000000       -3.4015        0.5000   2      7      4  dn",
        )

    def test_blocks_hold_the_occupied_or_all_states_of_each_channel(self) -> None:
        up = EigenChannel(np.array([-1.0, -0.5, 0.2]), np.array([1.0, 1.0, 0.0]), np.ones(3))
        down = EigenChannel(np.array([-0.9, -0.1, 0.3]), np.array([1.0, 0.0, 0.0]), np.ones(3))
        lines = eigen_dat_block((up, down), move=2, all_states=False)
        self.assertEqual([line[-2:] for line in lines], ["up", "up", "dn"])
        self.assertTrue(all(line[-9:-4] == "    2" for line in lines))
        self.assertEqual(len(eigen_dat_block((up,), move=0, all_states=True)), 3)
        k_lines = eigen_dat_block(
            (EigenChannel(np.r_[up.eigenvalues, up.eigenvalues], np.r_[up.occupations, up.occupations], ()),),
            move=0,
            all_states=False,
            kpoint_count=2,
        )
        self.assertEqual([int(line.split()[5]) for line in k_lines], [1, 1, 2, 2])


class CubeTests(unittest.TestCase):
    def test_fortran_exponential_format(self) -> None:
        self.assertEqual(fortran_exponential(0.0), "  0.00000E+00")
        self.assertEqual(fortran_exponential(6.552e-08), "  0.65520E-07")
        self.assertEqual(fortran_exponential(-1.52234), " -0.15223E+01")
        self.assertEqual(fortran_exponential(2.528e-319), "  0.25280-318")
        self.assertEqual(fortran_exponential(0.999996), "  0.10000E+01")

    def test_cube_header_and_box_of_a_cluster(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.57, radius=10.4, expansion_order=8))
        dense, origin, voxels = dense_box(np.ones(grid.size), grid)
        self.assertEqual(dense.shape, (38, 38, 38))
        lines = cube_text("charge_den", dense, origin, voxels).splitlines()
        # The header PARSEC wrote for this grid.
        self.assertEqual(
            lines[:7],
            [
                " Cubefile created from PARSEC calculation",
                " Volumetric charge_den data",
                "    1  -10.545000  -10.545000  -10.545000",
                "   38    0.570000    0.000000    0.000000",
                "   38    0.000000    0.570000    0.000000",
                "   38    0.000000    0.000000    0.570000",
                "    1    1.000000    0.000000    0.000000    0.000000",
            ],
        )
        self.assertEqual(len(lines), 7 + 38 * 38 * 7)
        self.assertEqual(lines[7], "  0.00000E+00" * 6)
        self.assertEqual(float(np.sum(dense)), float(grid.size))

    def test_the_origin_of_a_periodic_cube_is_its_first_point(self) -> None:
        # export_grid_data.f90 l.151-153: the origin (shift + i) step of the first point, on the default grid (half a
        # step off the cell's origin) as on the Fortran-derived one; the value at dense[i, j, k] is that of the point
        # origin + (i, j, k) steps
        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        for label, settings, expected in (
            ("default", PeriodicGridSettings(spacing=0.5, expansion_order=4, shift=(0.5, 0.5, 0.5)), 0.25),
            ("default, Ignore_Symmetry", PeriodicGridSettings(spacing=0.5, expansion_order=4, shift=(0.0, 0.0, 0.0)), 0.0),
            ("Fortran-derived", PeriodicGridSettings(spacing=0.5, expansion_order=4, shift=(0.5, 0.5, 0.5),
                                                     fortran_routines=True), 0.25),
        ):
            with self.subTest(label):
                grid = build_periodic_grid(cell, settings)
                values = grid.coordinates[:, 0] + 10.0 * grid.coordinates[:, 1] + 100.0 * grid.coordinates[:, 2]
                dense, origin, voxels = dense_box(values, grid)
                np.testing.assert_allclose(origin, [expected] * 3, rtol=0, atol=1e-14)
                np.testing.assert_allclose(voxels, np.eye(3) * 0.5, rtol=0, atol=1e-14)
                index = np.array([3, 5, 7])
                point = origin + index @ voxels
                self.assertAlmostEqual(dense[tuple(index)], point[0] + 10.0 * point[1] + 100.0 * point[2], places=12)
                header = cube_text("charge_den", dense, origin, voxels).splitlines()[2]
                self.assertEqual(header, f"    1{expected:12.6f}{expected:12.6f}{expected:12.6f}")


class OutputFileTests(unittest.TestCase):
    def test_default_wavefunctions_give_way_above_the_limit(self) -> None:
        files = RunOutputFiles(OutputFileSettings(), ".", write=lambda line: None)
        self.assertTrue(files.plan_wavefunctions(1000, 10, 1))
        self.assertTrue(files.selection_has_states)
        small = RunOutputFiles(
            OutputFileSettings(), ".", write=lambda line: None, wavefunction_limit_bytes=1000
        )
        self.assertFalse(small.plan_wavefunctions(1000, 10, 1))
        self.assertEqual(small.selection.indices, ())
        explicit = RunOutputFiles(
            OutputFileSettings(wavefunctions=WavefunctionSelection(mode="all", explicit=True)),
            ".",
            write=lambda line: None,
            wavefunction_limit_bytes=1000,
        )
        self.assertTrue(explicit.plan_wavefunctions(1000, 10, 1))
        off = RunOutputFiles(OutputFileSettings(output_data=False), ".", write=lambda line: None)
        self.assertFalse(off.plan_wavefunctions(1000, 10, 1))
        self.assertEqual(off.paths(), [])
        # the k-points of a k-point run count: 1000 points x 10 states x 16 bytes x 8 k-points = 1.28 MB
        counted = RunOutputFiles(OutputFileSettings(), ".", write=lambda line: None, wavefunction_limit_bytes=200_000)
        self.assertTrue(counted.plan_wavefunctions(1000, 10, 1, complex_values=True))
        self.assertFalse(counted.plan_wavefunctions(1000, 10, 1, complex_values=True, kpoint_count=8))

    def test_kpoint_loops_keep_their_orbitals_for_the_file_only(self) -> None:
        from parsec_python.cli import _orbitals_for_file_only, _parsec_dat_keep
        from parsec_python.Output.output_files import parsec_dat_kept_vectors as _parsec_dat_kept_vectors
        from parsec_python.models import SpinPolarizedSinglePointResult

        files = RunOutputFiles(OutputFileSettings(), ".", write=lambda line: None)
        files.plan_wavefunctions(1000, 10, 1)
        keep = _parsec_dat_keep(files, {})
        self.assertEqual(sorted(keep), ["keep_vectors", "keep_wavefunctions"])
        # a loop that keeps them already, a file without wavefunctions, no files, and the switch off: nothing added
        self.assertEqual(_parsec_dat_keep(files, {"keep_wavefunctions": True}), {})
        self.assertEqual(_parsec_dat_keep(None, {}), {})
        empty = RunOutputFiles(OutputFileSettings(output_data=False), ".", write=lambda line: None)
        empty.plan_wavefunctions(1000, 10, 1)
        self.assertEqual(_parsec_dat_keep(empty, {}), {})
        with patch.dict("os.environ", {dat.KPOINT_WAVEFUNCTIONS_SWITCH: "off"}):
            self.assertEqual(_parsec_dat_keep(files, {}), {})

        class Solver:
            def __init__(self, vectors):
                self.held = vectors

            def vectors(self, task):
                if self.held is None:
                    raise NotImplementedError
                return self.held

        np.testing.assert_array_equal(_parsec_dat_kept_vectors(Solver(np.eye(3)), 0), np.eye(3))
        # a Save_Wfn_Bands list keeps the lowest columns up to its highest state (a copy: the whole block is freed)
        whole = np.arange(12.0).reshape(4, 3)
        lowest = _parsec_dat_kept_vectors(Solver(whole), 0, columns=2)
        np.testing.assert_array_equal(lowest, whole[:, :2])
        self.assertIsNone(lowest.base)
        listed = RunOutputFiles(
            OutputFileSettings(wavefunctions=WavefunctionSelection(mode="list", indices=(1, 3), explicit=True)),
            ".", write=lambda line: None,
        )
        listed.plan_wavefunctions(1000, 10, 1)
        self.assertEqual(listed.kpoint_keep()["keep_vectors"].keywords, {"columns": 3})
        self.assertEqual(listed.kpoint_keep(whole=True)["keep_vectors"].keywords, {"columns": None})
        self.assertEqual(files.kpoint_keep()["keep_vectors"].keywords, {"columns": None})
        # the MPI-distributed solvers: an empty block, or none
        self.assertIsNone(_parsec_dat_kept_vectors(Solver(np.empty((3, 0))), 0))
        self.assertIsNone(_parsec_dat_kept_vectors(Solver(None), 0))
        # the fields keep_wavefunctions adds besides the orbitals are dropped
        fields = {name: None for name in SpinPolarizedSinglePointResult.__dataclass_fields__}
        result = SpinPolarizedSinglePointResult(**{
            **fields, "k_wavefunctions": ((np.eye(2),),), "next_potentials": (np.ones(2), np.ones(2)),
            "scf_input_potentials": (np.ones(2), np.ones(2)), "xc_potentials": (np.ones(2), np.ones(2)),
            "scf_hartree_potential": np.ones(2),
        })
        kept = _orbitals_for_file_only(result, True)
        self.assertIs(kept.k_wavefunctions, result.k_wavefunctions)
        for name in ("next_potentials", "scf_input_potentials", "xc_potentials", "scf_hartree_potential"):
            self.assertIsNone(getattr(kept, name))
        self.assertIs(_orbitals_for_file_only(result, False), result)

    def test_header_lines_only_for_keywords_of_the_input(self) -> None:
        self.assertEqual(output_header_lines(OutputFileSettings()), [])
        settings = OutputFileSettings(
            wavefunctions=WavefunctionSelection(mode="list", indices=tuple(range(1, 12)), explicit=True),
            eigenvalues=True,
            all_states=False,
            keywords=frozenset({"output_eigenvalues", "save_wfn_bands"}),
        )
        lines = output_header_lines(settings)
        self.assertEqual(lines[0], " Writing           11 selected wave functions to parsec.dat file:")
        self.assertEqual(lines[1], "".join(f"{n:6d}" for n in range(1, 11)))
        self.assertEqual(lines[2], "    11")
        self.assertIn(" outevflag =            1", lines)


if __name__ == "__main__":
    unittest.main()
