"""The full grid of a spin-polarized sector run whose symmetry guard fires (``PARSEC_SPIN_FALLBACK_SHARED``).

* CPU: the switch, and the plan (``driver.SpinFallbackPlan``): the estimate of the full grid of both channels against
  the memory of one device, the layout (``device`` as before, ``shared``: one sector of the identity whose basis the
  GPUs of the process or the ranks of the runner share), whether it fits, the NOTE of the preparation.
* CuPy (two GPUs or more for the shared layout): the O atom of the spin fixtures (a cluster) and in a periodic cube
  leave their sectors; the shared layout gives the energies, levels and moments of ``Spin_Symmetry_Sectors: off`` on
  the full grid of one device within the SCF tolerance, and a later loop of the same system (an ionic step) runs on
  it; the forces of a cell after the fallback (a single point: as they are; ``operations``: averaged over them); a
  plan that does not fit stops the run with an error before the full grid is prepared, after its NOTE.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available

_GIB = 2**30


def _plan(**changes):
    from parsec_python.acceleration.driver import SpinFallbackPlan

    values = dict(
        basis_bytes=int(21.8 * _GIB), device_bytes=80 * _GIB, devices=4, divided=True, shareable=True,
        setting="auto", mpi=False,
    )
    values.update(changes)
    return SpinFallbackPlan(**values)


class PlanTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import spin_fallback_shared_setting

        for value, expected in (("", "auto"), ("auto", "auto"), ("on", "on"), ("1", "on"), ("off", "off"),
                                ("no", "off"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": value}):
                self.assertEqual(spin_fallback_shared_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": "shared"}), self.assertRaises(ValueError):
            spin_fallback_shared_setting()

    def test_one_rank_per_node_shares_on_the_first_gpus(self) -> None:
        # The runner launched with one rank per node of 4 GPUs: the shared full grid of a fallback holds each rank's
        # share on its first GPU (the layout PARSEC_SINGLE_SECTOR_NODE_LAYOUT refuses for runs without a group); the
        # fallback keeps it and the NOTE of the preparation names it and advises --rank-per-gpu;
        # where nested sharing applies (PARSEC_MPI_NESTED_SHARING, the default auto) the shares spread over the
        # GPUs of every node instead (test_runner_fixes.NestedLayoutTests), so this is the plan under off
        off = mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "off"})
        off.start()
        self.addCleanup(off.stop)
        nodes = _plan(mpi=True, devices=2, rank_devices=4, gpu_per_rank=False)
        self.assertTrue(nodes.first_gpu_shares)
        self.assertEqual((nodes.layout, nodes.fits), ("shared", True))
        note = nodes.note()
        self.assertIn("each rank holds its share on the first of its 4 GPUs (one rank per node)", note)
        self.assertIn("2 of the 8 GPUs hold the basis", note)
        self.assertIn("--rank-per-gpu", note)
        self.assertIn("each on the first GPU of its node", nodes.layout_text())
        # under on the shared layout is taken where one device would hold it too: the NOTE still says so
        forced = _plan(mpi=True, devices=2, rank_devices=4, gpu_per_rank=False, basis_bytes=6 * _GIB, setting="on")
        self.assertTrue(forced.fits_one_device)
        self.assertIn("--rank-per-gpu", forced.note())
        # one GPU per rank, or one process: no such words, the NOTE of before
        for plan in (
            _plan(mpi=True, devices=8, rank_devices=1, gpu_per_rank=True),
            _plan(mpi=True, devices=2, rank_devices=4, gpu_per_rank=True),
            _plan(),
        ):
            with self.subTest(mpi=plan.mpi, gpu_per_rank=plan.gpu_per_rank):
                self.assertFalse(plan.first_gpu_shares)
                self.assertNotIn("--rank-per-gpu", plan.note())
                self.assertNotIn("first GPU of its node", plan.layout_text())
        self.assertIsNone(_plan(mpi=True, devices=8, gpu_per_rank=True, basis_bytes=6 * _GIB, setting="on").note())

    def test_the_layouts(self) -> None:
        # the neutral vacancy in 5^3 diamond cubes: 21.8 GiB of orbitals, 87 GiB estimated, over one A100 of 80 GiB
        large = _plan()
        self.assertFalse(large.fits_one_device)
        self.assertEqual((large.layout, large.fits), ("shared", True))
        self.assertIn("shared by the 4 GPUs of this process", large.note())
        self.assertIn("the 4 ranks of the runner", _plan(mpi=True).note())
        # what fits one device keeps the layout of before under auto, and has no NOTE
        small = _plan(basis_bytes=6 * _GIB)
        self.assertEqual((small.layout, small.fits, small.note()), ("device", True, None))
        self.assertEqual(_plan(basis_bytes=6 * _GIB, setting="on").layout, "shared")
        # a hybrid or meta-GGA, one GPU, a first solve that does not divide the basis: one device, which fails, and
        # the fallback stops before it (auto, on)
        for changes, why in (
            (dict(shareable=False), "hybrid or meta-GGA"),
            (dict(devices=1), "one GPU"),
            (dict(divided=False), "other than CHEBFF"),
        ):
            with self.subTest(why=why):
                plan = _plan(**changes)
                self.assertEqual((plan.layout, plan.fits, plan.refuses), ("device", False, True))
                self.assertIn(why, plan.note())
                self.assertIn("stops with an error", plan.note())
                self.assertEqual(plan.layout_text(), "none")
        # off: the full grid of one device is tried whatever the estimate, as before (no early error)
        off = _plan(setting="off")
        self.assertEqual((off.layout, off.fits, off.refuses), ("device", False, False))
        self.assertIn("PARSEC_SPIN_FALLBACK_SHARED=off", off.note())
        self.assertNotIn("stops with an error", off.note())
        self.assertIn("it is tried anyway", off.note())
        self.assertIn("one device (Spin_Symmetry_Sectors: off)", off.layout_text())
        self.assertEqual(_plan(shareable=False, setting="on").layout, "device")
        # four shares that do not fit either
        huge = _plan(basis_bytes=90 * _GIB)
        self.assertEqual((huge.layout, huge.fits, huge.refuses), ("shared", False, True))
        self.assertIn("does not fit either", huge.note())
        self.assertIn("4 GPUs of this process share", large.layout_text())
        self.assertIn("4 ranks of the runner share", _plan(mpi=True).layout_text())
        # no device to ask: nothing is refused
        self.assertTrue(_plan(device_bytes=None).fits)

    def test_the_note_names_the_layout(self) -> None:
        # parsec.out: where the full grid of a fallback goes, and that the forces come from there
        from parsec_python.models import SpinSectorFallbackRecord
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines

        def text(**changes):
            record = SpinSectorFallbackRecord(stage="the SCF loop", largest_fractional_occupation=0.5,
                                              tolerance=1e-3, **changes)
            return "\n".join(spin_sector_fallback_lines(record))

        self.assertIn("starts again on the full grid (Spin_Symmetry_Sectors: off).", text())
        shared = text(layout=_plan().layout_text())
        self.assertIn("starts again as one sector of the identity whose basis the 4 GPUs of this process share", shared)
        self.assertIn("its forces and any later structure are computed there", shared)
        self.assertIn("on the full grid of one device", text(layout=_plan(setting="off").layout_text()))
        self.assertIn("fits no device here, and the run stops", text(layout=_plan(devices=1).layout_text()))

    def test_the_plan_of_a_cell_on_the_full_grid(self) -> None:
        # Molecular dynamics, manual movements, Move_Flag, Ignore_Symmetry of a spin cell (moves): one device where
        # it fits, the shared layout where not, the k-point driver where neither (moves_note)
        self.assertIsNone(_plan(basis_bytes=6 * _GIB, moves=True).moves_note())
        shared = _plan(moves=True)
        self.assertEqual((shared.layout, shared.fits), ("shared", True))
        self.assertIn("molecular dynamics", shared.moves_note())
        self.assertIn("shared by the 4 GPUs of this process", shared.moves_note())
        self.assertNotIn("k-point driver", shared.moves_note())
        for changes in (dict(setting="off"), dict(devices=1), dict(divided=False), dict(basis_bytes=90 * _GIB)):
            with self.subTest(changes=changes):
                plan = _plan(moves=True, **changes)
                self.assertFalse(plan.fits)
                self.assertIn("keeps the 1x1x1 k-point driver", plan.moves_note())
        # the shared layout of a small cell under on (tests)
        self.assertEqual(_plan(basis_bytes=6 * _GIB, setting="on", moves=True).layout, "shared")

    def test_the_estimate_from_the_input(self) -> None:
        # driver.spin_full_grid_plan: the grid of the default routines, States_Num and the subspace buffer, both
        # channels in FP64, from the input alone (O2 in a 9.9 bohr cube on the 0.55 bohr grid: 18^3 points)
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_gamma_spin_sectors import _o2_cell

        directory, translation = _o2_cell(extra="Use_Fortran_Routines: .false.\n")
        self.addCleanup(shutil.rmtree, directory, True)
        problem = translation.problem
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": "auto"}):
            plan = driver.spin_full_grid_plan(problem)
        points = 18**3
        states = int(problem.scf.number_of_states) + int(problem.eigensolver.subspace_buffer)
        self.assertEqual(plan.basis_bytes, 2 * points * states * 8)
        self.assertTrue(plan.moves)
        # without States_Num: half the valence electrons (12 of O2) and six
        unset = driver.spin_full_grid_plan(replace(problem, scf=replace(problem.scf, number_of_states=None)))
        self.assertEqual(unset.basis_bytes, 2 * points * (6 + 6 + int(problem.eigensolver.subspace_buffer)) * 8)
        unpolarized = replace(problem, scf=replace(problem.scf, spin_polarized=False))
        self.assertIsNone(driver.spin_full_grid_plan(unpolarized))


def _two_gpus() -> bool:
    if not (cupy_available() and native_available()):
        return False
    import cupy as cp

    return cp.cuda.runtime.getDeviceCount() >= 2


def _cluster(directory: Path, name: str, mode: str):
    """A cluster of the spin fixtures with ``Spin_Symmetry_Sectors: mode``: ``o_spherical_ca`` (the O atom, CA,
    integer filling: the guard fires at the start), ``o2_ca`` (the O2 triplet, which stays in its sectors)."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.spin_reference import load_cases, write_case

    case = dict(load_cases()[name])
    if name == "o_spherical_ca":
        case["input"] = [line for line in case["input"] if not line.startswith("Fermi_Temp")]
        case.pop("occup", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(write_case(case, directory, extra=(f"Spin_Symmetry_Sectors: {mode}",))).problem


def _run(problem, records=None, *, symmetry="auto", warm_start=None, **options):
    from parsec_python.acceleration import driver

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = driver.prepare_single_point(problem, backend="auto", symmetry=symmetry, **options)
        system.materialize_final_wavefunctions = True
        result = driver.run_scf(
            system, callback=None if records is None else records.append, warm_start=warm_start
        )
    return system, result


def _warm_start(result):
    """The potentials of a converged full-grid run: a loop from them finds that state again (an ionic step of
    zero length), whichever broken-symmetry state of a partly filled shell the run from the atoms found."""

    from parsec_python.SCF import SCFWarmStart

    return SCFWarmStart(
        input_potential=tuple(np.asarray(channel.next_effective_potential) for channel in result.spin),
        hartree_potential=np.asarray(result.hartree_potential),
        eigensolver_state=None,
    )


@unittest.skipUnless(_two_gpus(), "two CUDA devices and the native extension are needed")
class SharedFallbackRunTests(unittest.TestCase):
    def directory(self) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="parsec_spin_fallback_"))
        self.addCleanup(shutil.rmtree, directory, True)
        return directory

    def assert_off_state(self, shared, full, *, energy=2e-8, levels=2e-7) -> None:
        self.assertTrue(shared.converged and full.converged)
        self.assertAlmostEqual(shared.energies.total, full.energies.total, delta=energy)
        for index in range(2):
            count = min(len(shared.spin[index].eigenvalues), len(full.spin[index].eigenvalues))
            np.testing.assert_allclose(
                shared.spin[index].eigenvalues[:count], full.spin[index].eigenvalues[:count], atol=levels
            )
            np.testing.assert_allclose(
                np.asarray(shared.spin[index].occupations)[:count], np.asarray(full.spin[index].occupations)[:count],
                atol=1e-6,
            )

    def test_the_shared_layout_is_the_full_grid(self) -> None:
        # The O2 triplet (one ground state): both channels in one sector of the identity shared by the GPUs of the
        # process against Spin_Symmetry_Sectors: off on the full grid of one device.
        _, full = _run(_cluster(self.directory(), "o2_ca", "off"))
        # (the basis divided over the GPUs whatever its size: PARSEC_CUPY_DISTRIBUTED_STATE=on)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_DISTRIBUTED_STATE": "on"}):
            system, shared = _run(
                _cluster(self.directory(), "o2_ca", "auto"), symmetry="off", spin_identity_sector=True
            )
        details = dict(shared.backend.details)
        self.assertIn("one sector of the identity, the full grid", details["spin_symmetry_sectors"])
        self.assertIn("for both spin channels", details["symmetry_detection"])
        solver = system.backend.symmetry_eigensolver
        self.assertEqual(solver.decomposition.representation_count, 1)
        self.assertGreaterEqual(len(solver.sector_device_groups[0]), 2)
        self.assertIsNone(system.spin_sector_fallback)
        self.assert_off_state(shared, full, energy=1e-8)
        for index in range(2):
            np.testing.assert_allclose(shared.spin[index].density, full.spin[index].density, atol=1e-7)
        # the orbitals of the divided basis read from the devices that hold them (the result's host matrix)
        volume = float(system.reference.grid.volume_element)
        for index in range(2):
            orbitals = np.asarray(shared.spin[index].wavefunctions)
            np.testing.assert_allclose(np.sum(orbitals * orbitals, axis=0), 1.0, atol=1e-10)
            occupations = np.asarray(shared.spin[index].occupations, dtype=np.float64)
            density = (orbitals[:, : occupations.size] ** 2) @ occupations / volume
            np.testing.assert_allclose(density, shared.spin[index].density, atol=1e-8)

    def test_the_oxygen_atom_leaves_its_sectors_for_the_shared_layout(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.models import SpinSectorFallbackRecord

        _, full = _run(_cluster(self.directory(), "o_spherical_ca", "off"))
        records = []
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": "on"}):
            system, shared = _run(
                _cluster(self.directory(), "o_spherical_ca", "auto"), records, warm_start=_warm_start(full)
            )
        # (with PARSEC_ORBITAL_GROUP_SPIN=auto: the O atom of Oh takes the weighted sectors first, whose guard
        # sends it to the abelian ones (a record of route "weighted"); one record of the full grid follows)
        fallbacks = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len([record for record in fallbacks if getattr(record, "route", "spin") != "weighted"]), 1)
        self.assertLessEqual(len([record for record in fallbacks if getattr(record, "route", "spin") == "weighted"]), 1)
        details = dict(shared.backend.details)
        self.assertIn("left the sectors", details["spin_symmetry_sectors"])
        self.assertIn("one sector of the identity shared by", details["spin_symmetry_sectors"])
        self.assertIsNone(system.spin_sector_fallback)
        self.assertEqual(system.backend.symmetry_eigensolver.decomposition.representation_count, 1)
        self.assert_off_state(shared, full, energy=1e-7, levels=1e-6)
        # a later loop of the same system (the next structure of a relaxation) runs on the shared full grid
        warm = driver.warm_start_for_moved_atoms(system, shared, system)
        again = driver.run_scf(system, warm_start=warm)
        self.assertTrue(again.converged)
        self.assertAlmostEqual(again.energies.total, full.energies.total, delta=1e-7)
        # auto keeps the layout of before where the full grid fits one device: Spin_Symmetry_Sectors: off bit for bit
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": "auto"}):
            _, kept = _run(_cluster(self.directory(), "o_spherical_ca", "auto"))
        self.assertEqual(kept.energies.total, full.energies.total)
        self.assertNotIn("shared by", dict(kept.backend.details)["spin_symmetry_sectors"])

    def test_a_cell_shares_its_full_grid_and_has_forces(self) -> None:
        # The O atom at the origin of a periodic cube (D2h sectors; the guard fires); its forces vanish (one atom of a
        # crystal), and they are those of the full grid without a symmetrization the cell no longer has.
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_kpoint_sector_guard import _oxygen

        directory, problem = _oxygen("80", criterion="1.0e-8")
        self.addCleanup(shutil.rmtree, directory, True)
        _, full = _run(replace(problem, scf=replace(problem.scf, spin_symmetry_sectors="off")))
        results = {}
        for setting in ("on", "off"):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": setting}):
                system, result = _run(problem, warm_start=None if setting == "off" else _warm_start(full))
                details = dict(result.backend.details)
                self.assertIn("left the sectors", details["spin_symmetry_sectors"])
                self.assertEqual("shared by" in details["spin_symmetry_sectors"], setting == "on")
                # a single point after the fallback: the forces of the full grid as they are; a relaxation's
                # operations average them
                single = driver.compute_forces(system, result, symmetrize=True)
                results[setting] = (result, np.asarray(single.total))
        # (the top level of the subspace converges least)
        self.assert_off_state(results["on"][0], full, levels=1e-6)
        self.assertEqual(results["off"][0].energies.total, full.energies.total)
        np.testing.assert_allclose(results["on"][1], results["off"][1], rtol=0, atol=1e-6)
        self.assertLess(float(np.max(np.abs(results["off"][1]))), 1e-3)

    def test_a_cell_is_prepared_as_the_identity_sector_of_the_ranks(self) -> None:
        # The runner's ranks prepare a spin cell as one sector of the identity of both channels (after its guard
        # fired, PARSEC_SPIN_FALLBACK_SHARED, or for moves without a point group, mpi_full_scf._spin_moves_without_group):
        # before 2026-11-21 the early refusal of spin cells without a group in the driver stopped that preparation
        # ("it is prepared without symmetry"; the runner's shared fallback of a cell on 4 ranks failed so on A100).
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext
        from parsec_python.acceleration.tests.test_kpoint_sector_guard import _oxygen
        from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World

        directory, problem = _oxygen("80", criterion="1.0e-8")
        self.addCleanup(shutil.rmtree, directory, True)
        context = MPISectorContext(_Comm(_World(1), 0))
        context.divided_bases = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="auto", symmetry="off", mpi_context=context,
                                                 spin_identity_sector=True)
        solver = system.backend.symmetry_eigensolver
        self.assertEqual(solver.decomposition.representation_count, 1)
        self.assertIn("for both spin channels", dict(system.backend_info.details)["symmetry_detection"])
        self.assertIsNone(system.spin_sector_fallback)
        # without the identity sector the cell is still refused before its preparation, as before
        with warnings.catch_warnings(), self.assertRaisesRegex(ValueError, "it is prepared without symmetry"):
            warnings.simplefilter("ignore")
            driver.prepare_single_point(problem, backend="auto", symmetry="off",
                                        mpi_context=MPISectorContext(_Comm(_World(1), 0)))

    def test_a_plan_that_does_not_fit_stops_before_the_full_grid(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_kpoint_sector_guard import _oxygen

        directory, problem = _oxygen("80", criterion="1.0e-8")
        self.addCleanup(shutil.rmtree, directory, True)
        original = driver.spin_fallback_plan

        def tiny(reference, **keywords):
            plan = original(reference, **keywords)
            plan.device_bytes = 1024  # a device of 1 kiB: neither layout fits
            return plan

        with mock.patch.object(driver, "spin_fallback_plan", tiny), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="auto", symmetry="auto")
            notes = [value for key, value in system.backend_info.details if key == "symmetry_full_grid_note"]
            self.assertTrue(any("should the symmetry guard of the spin sectors fire" in note for note in notes), notes)
            with self.assertRaisesRegex(ValueError, "has no full grid that fits"):
                driver.run_scf(system)
        # off: the NOTE, and the full grid of one device tried as before (it fits the real device)
        with (
            mock.patch.object(driver, "spin_fallback_plan", tiny),
            mock.patch.dict(os.environ, {"PARSEC_SPIN_FALLBACK_SHARED": "off"}),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="auto", symmetry="auto")
            notes = [value for key, value in system.backend_info.details if key == "symmetry_full_grid_note"]
            self.assertTrue(any("it is tried anyway" in note for note in notes), notes)
            records = []
            result = driver.run_scf(system, callback=records.append)
        self.assertTrue(result.converged)
        self.assertIn("left the sectors", dict(result.backend.details)["spin_symmetry_sectors"])
        from parsec_python.models import SpinSectorFallbackRecord
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines

        # (the full-grid record; a weighted run records its step to the abelian sectors before it, route "weighted")
        fallback = [
            record
            for record in records
            if isinstance(record, SpinSectorFallbackRecord) and getattr(record, "route", "spin") != "weighted"
        ]
        self.assertEqual(len(fallback), 1)
        self.assertIn("on the full grid of one device", "\n".join(spin_sector_fallback_lines(fallback[0])))


if __name__ == "__main__":
    unittest.main()
