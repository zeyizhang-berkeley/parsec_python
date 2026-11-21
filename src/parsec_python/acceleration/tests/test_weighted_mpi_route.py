"""Where spin and occup.in take the weighted sectors since 2026-11-21 (``Symmetry/weighted_inputs.py``).

Checked on the host: the three switches (``PARSEC_ORBITAL_GROUP_MPI``, ``PARSEC_WEIGHTED_GUARD_SHORTCUT``,
``PARSEC_ORBITAL_GROUP_SPIN_FLOOR``) and the refusals they give; which weighted sectors of Td and Oh the abelian
group of the atoms splits (``orbital_groups.abelian_split_sectors``) and the partner labels of the spin guard that
follow; that a guard stop of the weighted sectors says whether the abelian sectors would stop the same levels, and
that the fallback then goes straight to the full grid; the NOTE of parsec.out; the abelian sectors prepared again
on every MPI rank (``MPISectorContext.reprepare`` with ``abelian_reason``).  The GPU runs of the route are in
``test_weighted_inputs.py`` and in the A100 measurements of the switch table.
"""

from __future__ import annotations

from itertools import permutations, product
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from ..Symmetry.weighted_inputs import (
    GUARD_SHORTCUT_SWITCH,
    MPI_SWITCH,
    SPIN_FLOOR_ENTRIES,
    SPIN_FLOOR_SWITCH,
    abelian_orbitals,
    forced_abelian_reason,
    guard_shortcut_setting,
    mpi_weighted_setting,
    spin_floor_setting,
    weighted_route_settings,
)


def _signed_permutations():
    for order in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            yield matrix


_TETRAHEDRON = {(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)}
TD = [m for m in _signed_permutations() if {tuple(int(v) for v in m @ np.array(p)) for p in _TETRAHEDRON} == _TETRAHEDRON]
OH = list(_signed_permutations())


def _diagonal(operations):
    return [index for index, matrix in enumerate(operations) if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0]


def _problem(**scf):
    defaults = dict(
        spin_polarized=True,
        noncollinear=False,
        self_consistent_spin_orbit=False,
        is_hybrid=False,
        is_meta_gga=False,
        xc_functional="pbe",
        fixed_occupations=None,
        spin_symmetry_sector_mode="auto",
        symmetrize_density=False,
    )
    defaults.update(scf)
    return SimpleNamespace(
        scf=SimpleNamespace(**defaults), pseudopotentials={}, eigensolver=SimpleNamespace(fixed_state_count=False)
    )


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {MPI_SWITCH: "", GUARD_SHORTCUT_SWITCH: "", SPIN_FLOOR_SWITCH: ""}):
            self.assertEqual(weighted_route_settings(), ("auto", "auto", SPIN_FLOOR_ENTRIES))
        with mock.patch.dict(os.environ, {MPI_SWITCH: "OFF", GUARD_SHORTCUT_SWITCH: "off", SPIN_FLOOR_SWITCH: "off"}):
            self.assertEqual((mpi_weighted_setting(), guard_shortcut_setting(), spin_floor_setting()), ("off", "off", 0))
        with mock.patch.dict(os.environ, {SPIN_FLOOR_SWITCH: "12345"}):
            self.assertEqual(spin_floor_setting(), 12345)
        for name, value in ((MPI_SWITCH, "on"), (GUARD_SHORTCUT_SWITCH, "abelian"), (SPIN_FLOOR_SWITCH, "-1"),
                            (SPIN_FLOOR_SWITCH, "big")):
            with self.subTest(name=name, value=value), mock.patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    weighted_route_settings()

    def test_refusals(self) -> None:
        from .. import driver

        refusal = driver._orbital_group_refusal
        context = SimpleNamespace(rank=0, size=2, exchange=False)
        spin = _problem()
        fixed = _problem(spin_polarized=False, fixed_occupations=(2.0, 0.0))
        with mock.patch.dict(os.environ, {MPI_SWITCH: "auto", SPIN_FLOOR_SWITCH: "1000"}):
            # several ranks take the weighted sectors as one process does
            self.assertIsNone(refusal(spin, True, context))
            self.assertIsNone(refusal(fixed, False, context))
            # the floor: grid points x states below 1000 keep the abelian group, on one rank or several alike
            for where in (None, context):
                kept = refusal(spin, True, where, basis=999)
                self.assertIn(SPIN_FLOOR_SWITCH, kept)
                self.assertIsNone(refusal(spin, True, where, basis=1000))
                self.assertIsNone(refusal(spin, True, where))
                self.assertIn("keep the sectors of the abelian group", driver._kept_abelian_note(kept))
            # an unpolarized run has no floor
            self.assertIsNone(refusal(fixed, False, None, basis=1))
            # a run prepared again after it left the weighted sectors keeps the abelian group, with no NOTE of these
            with abelian_orbitals("left"):
                self.assertEqual(refusal(spin, True, context, basis=10**9), "left")
                self.assertIsNone(driver._kept_abelian_note(refusal(spin, True, context)))
        with mock.patch.dict(os.environ, {MPI_SWITCH: "off", SPIN_FLOOR_SWITCH: "off"}):
            for problem, polarized in ((spin, True), (fixed, False)):
                kept = refusal(problem, polarized, context)
                self.assertIn(MPI_SWITCH, kept)
                self.assertIsNotNone(driver._kept_abelian_note(kept))
                self.assertIsNone(refusal(problem, polarized, None, basis=1))
        self.assertIsNone(forced_abelian_reason())
        # the other reasons of before give no NOTE
        self.assertIsNone(driver._kept_abelian_note("a hybrid functional"))
        self.assertIsNone(driver._kept_abelian_note(None))

    def test_floor_basis(self) -> None:
        from .. import driver

        problem = _problem(number_of_states=12)
        self.assertEqual(driver._spin_floor_basis(problem, SimpleNamespace(size=1000)), 12000)
        self.assertIsNone(driver._spin_floor_basis(problem, SimpleNamespace()))


class AbelianSplitTests(unittest.TestCase):
    def _plan(self, operations):
        from ..Symmetry.orbital_groups import abelian_split_sectors, multiplication_table, orbital_group_plan

        plan = orbital_group_plan(operations)
        table = multiplication_table(operations)
        return plan, abelian_split_sectors(table, plan, _diagonal(operations))

    def test_td_over_d2(self) -> None:
        # D2d: the sectors A1 and B1 hold a1, a2 and the partners of e, which lie in A of D2 alone; A2 and B2 hold
        # one partner of every t1 and t2, which split over B1, B2, B3 of D2
        self.assertEqual(len(TD), 24)
        plan, split = self._plan(TD)
        self.assertEqual(sorted(plan.weights), [1, 1, 3, 3])
        self.assertEqual(list(split), [weight == 3 for weight in plan.weights])

    def test_oh_over_d2h(self) -> None:
        plan, split = self._plan(OH)
        self.assertEqual(len(split), len(plan.weights))
        self.assertEqual(list(split), [weight == 3 for weight in plan.weights])
        self.assertIn(True, split)
        self.assertIn(False, split)

    def test_reduction_of_a_cluster(self) -> None:
        # CH4 (Td) on a small grid: the weighted sectors of D2d carry the flags of the abelian group D2 it was found
        # from; the sectors of weight 3 hold the t1 and t2 partners, which D2 splits
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import Atom, GridSettings

        from ..Symmetry import PointGroupReduction, SignedPermutationReduction, detect_point_group_reduction

        atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
            Atom("H", tuple(1.2 * np.array(signs, dtype=float)))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.8, expansion_order=4, shift=(0.5,) * 3))
        reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertFalse(reduction.sector_abelian_split.flags.writeable)
        self.assertEqual(reduction.sector_abelian_split.tolist(), (reduction.sector_weights == 3).tolist())

    def test_reduction_of_a_cell(self) -> None:
        # A cell whose operations take no translation (the neutral vacancy at the origin of a 2^3 diamond cell, Td)
        # keeps the abelian group about the origin when it leaves the weighted sectors (periodic_abelian_reduction:
        # no larger group of commuting involutions), so its sectors carry the flags of D2, as a cluster's do.  The
        # pristine cell has translations: its abelian sectors take them, and the flags are not known.
        from ..Symmetry import PointGroupReduction, SignedPermutationReduction, detect_point_group_reduction
        from ..Symmetry import axis_reflection
        from ..Symmetry.axis_reflection import periodic_abelian_reduction
        from .test_gamma_spin_sectors import _vacancy_cell

        grid, atoms = _vacancy_cell(2, (0.0, 0.0, 0.0), points_per_cube=8)
        origin = SignedPermutationReduction.detect(grid, atoms)
        self.assertEqual(origin.group_order, 4)
        reduction = detect_point_group_reduction(grid, atoms, origin)
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertIsNone(getattr(reduction, "translations", None))
        self.assertEqual(sorted(reduction.sector_weights.tolist()), [1, 1, 3, 3])
        self.assertEqual(reduction.sector_abelian_split.tolist(), (reduction.sector_weights == 3).tolist())
        self.assertIs(periodic_abelian_reduction(grid, atoms, origin), origin)
        # were there a larger group of commuting involutions about the origin, the cell would take it: no flags
        with mock.patch.object(axis_reflection, "_abelian_is_largest_involution_group", return_value=False):
            self.assertIsNone(detect_point_group_reduction(grid, atoms, origin).sector_abelian_split)
        # PARSEC_PERIODIC_ABELIAN_TRANSLATIONS=off keeps the group about the origin in any case
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_ABELIAN_TRANSLATIONS": "off"}), mock.patch.object(
            axis_reflection, "_abelian_is_largest_involution_group", return_value=False
        ):
            flags = detect_point_group_reduction(grid, atoms, origin).sector_abelian_split
            self.assertEqual(flags.tolist(), (reduction.sector_weights == 3).tolist())
        pristine_grid, pristine = _vacancy_cell(2, None, points_per_cube=8)
        weighted = detect_point_group_reduction(
            pristine_grid, pristine, SignedPermutationReduction.detect(pristine_grid, pristine)
        )
        self.assertIsInstance(weighted, PointGroupReduction)
        self.assertIsNotNone(weighted.translations)
        self.assertIsNone(weighted.sector_abelian_split)

    def test_abelian_group_itself(self) -> None:
        # the group of the abelian route splits nothing it holds (every representation is one-dimensional)
        from ..Symmetry.orbital_groups import abelian_split_sectors, multiplication_table, orbital_group_plan

        d2h = [OH[index] for index in _diagonal(OH)]
        plan = orbital_group_plan(d2h)
        self.assertEqual(abelian_split_sectors(multiplication_table(d2h), plan, range(8)), (False,) * 8)

    def test_partner_labels(self) -> None:
        from ..SCF.spin_sectors import PARTNER_LABEL_STRIDE, abelian_partner_labels

        labels = abelian_partner_labels([1, 4, 2], [1, 3, 1], [False, False, False, True])
        np.testing.assert_array_equal(labels, [0, *(4 * PARTNER_LABEL_STRIDE + np.arange(3) + 1), 0])
        self.assertIsNone(abelian_partner_labels([1], [1], None))
        with self.assertRaises(ValueError):
            abelian_partner_labels([5], [1], [True, True])


class GuardTests(unittest.TestCase):
    """The guard of D2d sectors (weights 1, 1, 3, 3; the last two split by D2) says what the abelian sectors see."""

    def _adapter(self, split=(False, False, True, True)):
        from ..SCF.spin_sectors import SpinSectorReducer

        reduction = SimpleNamespace(
            sector_weights=np.array([1, 1, 3, 3]), sector_abelian_split=None if split is None else np.array(split),
            wedge_size=4, multiplicities=np.ones(4),
        )
        return SpinSectorReducer(SimpleNamespace(reduction=reduction, weights=np.ones(4)), hybrid=False)

    def test_start_report(self) -> None:
        levels = [np.array([-1.0, -0.5, -0.5, 0.2]), np.array([-1.0, -0.5, -0.5, 0.2])]
        sectors = [np.array([1, 4, 2, 3]), np.array([1, 4, 2, 3])]
        # one electron in a t2 level (a state of weight 3 in sector 4, split by D2) of the down channel
        report = self._adapter().start_spectrum_report(
            levels, [np.array([1.0, 3.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])],
            [np.array([1, 4, 1, 3]), np.array([1, 4, 1, 3])],
        )
        self.assertIsNotNone(report)
        self.assertTrue(report.abelian)
        # one electron in an e level (sectors 1 and 2 of weight 1, one sector of D2): the abelian sectors may hold it
        e_levels = [np.array([-1.0, -0.5, -0.2, -0.2]), np.array([-1.0, -0.5, -0.2, -0.2])]
        report = self._adapter().start_spectrum_report(
            e_levels, [np.array([1.0, 3.0, 1.0, 1.0]), np.array([1.0, 3.0, 1.0, 0.0])],
            [np.array([1, 4, 1, 2]), np.array([1, 4, 1, 2])],
        )
        self.assertIsNotNone(report)
        self.assertFalse(report.abelian)
        # not known (a cell): no shortcut
        report = self._adapter(split=None).start_spectrum_report(
            levels, [np.array([1.0, 3.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])],
            [np.array([1, 4, 1, 3]), np.array([1, 4, 1, 3])],
        )
        self.assertFalse(report.abelian)
        # the line of further solves keeps the flag
        from ..SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        with self.assertRaises(SpinSectorsAbandoned) as caught:
            self._adapter().abandon_start(
                FractionalOccupations(largest=0.3, tolerance=0.05, lines=("x",), abelian=True), passes=2
            )
        self.assertTrue(caught.exception.report.abelian)
        self.assertIn("2 further solve", caught.exception.report.lines[-1])

    def _channel(self, values, occupations, states):
        return SimpleNamespace(
            eigenvalues=np.repeat(values, states[2]),
            occupations=np.asarray(occupations, dtype=float),
            representations=np.repeat(states[0], states[2]),
            sector_states=states,
            density=np.zeros(4),
        )

    def test_end_report(self) -> None:
        from ..SCF.spin_sectors import SpinSectorsAbandoned

        adapter = self._adapter()
        # a fractional t2 (weight 3, sector 4) at the Fermi level: the abelian sectors stop it too
        states = (np.array([1, 4, 2]), np.array([1.0, 2.0, 0.0]), np.array([1, 3, 1]))
        t2 = self._channel([-1.0, -0.5, 0.2], [1.0, 2 / 3, 2 / 3, 2 / 3, 0.0], states)
        with self.assertRaises(SpinSectorsAbandoned) as caught:
            adapter.check_spin_result(SimpleNamespace(spin=(t2, t2)), final=True)
        self.assertTrue(caught.exception.report.abelian)
        # a fractional e (sectors 1 and 2, weight 1 each): one sector of D2, which the abelian sectors may hold
        states = (np.array([1, 2, 4]), np.array([0.5, 0.5, 0.0]), np.array([1, 1, 3]))
        e = self._channel([-0.5, -0.5, 0.2], [0.5, 0.5, 0.0, 0.0, 0.0], states)
        with self.assertRaises(SpinSectorsAbandoned) as caught:
            adapter.check_spin_result(SimpleNamespace(spin=(e, e)), final=True)
        self.assertFalse(caught.exception.report.abelian)
        # partners of an e filled unequally (one whole, one empty): only the weighted sectors refuse that
        states = (np.array([1, 2, 4]), np.array([1.0, 0.0, 0.0]), np.array([1, 1, 3]))
        unequal = self._channel([-0.5, -0.5, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0], states)
        with self.assertRaises(SpinSectorsAbandoned) as caught:
            adapter.check_spin_result(SimpleNamespace(spin=(unequal, unequal)), final=True)
        self.assertFalse(caught.exception.report.abelian)


class FallbackTests(unittest.TestCase):
    def _system(self, weighted: bool):
        from ..Symmetry import PointGroupReduction

        reduction = mock.MagicMock(spec=PointGroupReduction) if weighted else object()
        return SimpleNamespace(
            backend=SimpleNamespace(
                symmetry_eigensolver=SimpleNamespace(decomposition=SimpleNamespace(reduction=reduction))
            ),
            scalar_field_adapter=SimpleNamespace(check_spin_result=lambda result, final: None),
        )

    def _run(self, abelian: bool, setting: str):
        from .. import driver
        from ..SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        fallback = driver._SpinSectorFallback(
            SimpleNamespace(pseudopotentials={}), (), backend="auto", symmetry="auto", symmetry_cache_directory=None
        )
        replaced, records, calls = [], [], []

        def to_abelian(system, stage, report):
            replaced.append("abelian")
            system.backend = self._system(False).backend

        def to_full_grid(system, stage, report):
            replaced.append("full")
            system.backend = SimpleNamespace(symmetry_eigensolver=None)

        fallback.replace_with_abelian = to_abelian
        fallback.replace_with_full_grid = to_full_grid
        report = FractionalOccupations(largest=0.33, tolerance=0.05, lines=("x",), abelian=abelian)

        def run(system, callback=None, warm_start=None, **keywords):
            calls.append(system.backend.symmetry_eigensolver)
            if system.backend.symmetry_eigensolver is not None:
                raise SpinSectorsAbandoned("the SCF loop", report)
            return "result"

        with mock.patch.dict(os.environ, {GUARD_SHORTCUT_SWITCH: setting}):
            self.assertEqual(fallback.run(self._system(True), run, callback=records.append), "result")
        return replaced, records

    def test_shortcut(self) -> None:
        replaced, records = self._run(True, "auto")
        self.assertEqual(replaced, ["full"])
        self.assertEqual([(record.route, record.abelian_skipped) for record in records],
                         [("weighted", True), ("spin", False)])
        # off, or a report the abelian sectors may keep: through them, as before
        for abelian, setting in ((True, "off"), (False, "auto")):
            replaced, records = self._run(abelian, setting)
            self.assertEqual(replaced, ["abelian", "full"])
            self.assertEqual([(record.route, record.abelian_skipped) for record in records],
                             [("weighted", False), ("spin", False)])

    def test_note(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        record = SpinSectorFallbackRecord(stage="the SCF loop", largest_fractional_occupation=0.3, tolerance=0.05,
                                          route="weighted", abelian_skipped=True)
        text = "\n".join(spin_sector_fallback_lines(record))
        self.assertIn("straight to the full grid", text)
        self.assertIn(GUARD_SHORTCUT_SWITCH, text)
        text = "\n".join(spin_sector_fallback_lines(SpinSectorFallbackRecord(
            stage="the SCF loop", largest_fractional_occupation=0.3, tolerance=0.05, route="weighted")))
        self.assertIn("starts again in the sectors of the abelian group", text)


class _Comm:
    def __init__(self, rank=0, size=2):
        self.rank, self.size = rank, size
        self.sent = []

    def bcast(self, message, root=0):
        self.sent.append(message)
        return message


class MPIRepareTests(unittest.TestCase):
    def test_root_sends_the_abelian_input(self) -> None:
        from ..experimental.mpi_scf import MPISectorContext

        context = MPISectorContext(_Comm())
        context.reprepare(dict(problem="p", abelian_reason="left", backend="auto"))
        self.assertFalse(context.spin_fallback_layout)
        self.assertEqual(context.comm.sent[-1][0], "reprepare")
        context.reprepare(dict(problem="p", backend="auto"))
        self.assertTrue(context.spin_fallback_layout)

    def test_worker_prepares_the_abelian_sectors(self) -> None:
        from .. import driver
        from ..experimental.mpi_scf import MPISectorContext

        context = MPISectorContext(_Comm(rank=1))
        seen = {}

        def prepare(problem, **options):
            seen.update(options, reason=forced_abelian_reason(), problem=problem)
            return SimpleNamespace(backend=SimpleNamespace(symmetry_eigensolver="solver"), orbital_density_builder="b")

        solver = SimpleNamespace(restore_memory_allocator=lambda: None)
        with mock.patch.object(driver, "prepare_single_point", prepare):
            result = context._reprepare_worker(solver, dict(
                problem="p", abelian_reason="left after the SCF loop", backend="auto", symmetry="auto",
                symmetry_cache_directory=None, symmetry_fixed_atoms=None, symmetry_constraints=(), sector_forces=True,
            ))
        self.assertEqual(result, ("solver", "b"))
        self.assertEqual(seen["reason"], "left after the SCF loop")
        self.assertIs(seen["mpi_context"], context)
        self.assertTrue(seen["sector_forces"])
        self.assertNotIn("spin_identity_sector", seen)
        self.assertFalse(context.spin_fallback_layout)
        self.assertIsNone(forced_abelian_reason())

    def test_root_prepares_with_the_ranks(self) -> None:
        from .. import driver
        from ..experimental.mpi_scf import MPISymmetryDensityBuilder

        order = []
        context = SimpleNamespace(reprepare=lambda payload: order.append(("reprepare", payload)))

        def prepare(problem, **options):
            order.append(("prepare", options.get("mpi_context"), forced_abelian_reason()))
            return SimpleNamespace(
                backend=SimpleNamespace(symmetry_eigensolver="abelian solver", info=None),
                backend_info=SimpleNamespace(details=()),
                orbital_density_builder="local builder",
            )

        fallback = driver._SpinSectorFallback(
            SimpleNamespace(pseudopotentials={}, scf=SimpleNamespace()), (), backend="auto", symmetry="auto",
            symmetry_cache_directory=None, mpi_context=context, sector_forces=True,
        )
        fallback._problem_of = lambda system, **keywords: "problem"
        fallback._with_present_pseudopotentials = lambda problem, directory: problem
        resets = []
        system = SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(
                spin_channel_count=2, for_spin_channel=lambda index: SimpleNamespace(reset=lambda: resets.append(index)),
            )),
            materialize_final_wavefunctions=False,
        )
        report = SimpleNamespace(reason="fractional occupations", measure="m", largest=0.3)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))
        with mock.patch.object(driver, "prepare_single_point", prepare), \
                mock.patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(cupy, None)), \
                mock.patch.object(driver, "replace", lambda item, **changes: SimpleNamespace(
                    **{**vars(item), **changes})), \
                mock.patch("dataclasses.fields", lambda item: [SimpleNamespace(name=name) for name in vars(item)]):
            fallback.replace_with_abelian(system, "the SCF loop", report)
        self.assertEqual(resets, [0, 1])
        self.assertEqual([item[0] for item in order], ["reprepare", "prepare"])
        payload = order[0][1]
        self.assertEqual(payload["problem"], "problem")
        self.assertIn("the weighted sectors were left", payload["abelian_reason"])
        self.assertTrue(payload["sector_forces"])
        self.assertIs(order[1][1], context)
        self.assertIn("the weighted sectors were left", order[1][2])
        self.assertIsInstance(system.orbital_density_builder, MPISymmetryDensityBuilder)
        self.assertEqual(system.orbital_density_builder.local_builder, "local builder")
        self.assertEqual(system.backend.symmetry_eigensolver, "abelian solver")


def _device_available() -> bool:
    from ..backends.cupy import cupy_available
    from ..backends.native import native_available

    return cupy_available() and native_available()


@unittest.skipUnless(_device_available(), "CUDA runtime and native extension are unavailable")
class DeviceTests(unittest.TestCase):
    """GeH4 (Td, the weighted sectors of D2d, the abelian group D2) spin-polarized on a GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile
        from pathlib import Path

        from .test_weighted_inputs import _write_problem

        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.problem = _write_problem(Path(directory.name))

    def spin(self, charge: float):
        from dataclasses import replace

        problem = self.problem
        return replace(problem, scf=replace(problem.scf, spin_polarized=True, net_charge=charge))

    def _run(self, problem, environment):
        from parsec_python.acceleration import driver

        records = []
        with mock.patch.dict(os.environ, environment):
            system = driver.prepare_single_point(problem, symmetry="on")
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system, callback=records.append)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        return SimpleNamespace(
            system=system, result=result, records=records,
            reduction=None if sectors is None else sectors.decomposition.reduction,
            details=system.backend_info.details,
        )

    def test_the_floor_keeps_the_abelian_group_of_a_small_system(self) -> None:
        from ..Symmetry import PointGroupReduction

        problem = self.spin(0.0)
        # (the default floor is none, SPIN_FLOOR_ENTRIES = 0; a floor given as a number of entries)
        kept = self._run(problem, {SPIN_FLOOR_SWITCH: str(10**12)})
        self.assertNotIsInstance(kept.reduction, PointGroupReduction)
        self.assertEqual(int(kept.reduction.group_order), 4)
        notes = [value for key, value in kept.details if key == "symmetry_full_grid_note"]
        self.assertTrue(any(SPIN_FLOOR_SWITCH in note for note in notes), notes)
        abelian = self._run(problem, {"PARSEC_ORBITAL_GROUP_SPIN": "off"})
        # the route of PARSEC_ORBITAL_GROUP_SPIN=off, bit for bit
        self.assertEqual(float(kept.result.energies.total), float(abelian.result.energies.total))
        weighted = self._run(problem, {SPIN_FLOOR_SWITCH: "auto"})
        self.assertIsInstance(weighted.reduction, PointGroupReduction)
        self.assertEqual(weighted.reduction.sector_abelian_split.tolist(),
                         (weighted.reduction.sector_weights == 3).tolist())
        self.assertFalse(any(key == "symmetry_full_grid_note" and SPIN_FLOOR_SWITCH in value
                             for key, value in weighted.details))
        self.assertAlmostEqual(float(weighted.result.energies.total), float(abelian.result.energies.total),
                               delta=1e-7)

    def test_a_partly_filled_t2_goes_straight_to_the_full_grid(self) -> None:
        # The cation's t2 shell, a state of weight 3 in sector B2 of D2d: D2 splits it too, so the abelian
        # sectors would stop it as well; the run takes the full grid at once, the result of the abelian route
        # (which goes there after its own guard) bit for bit.
        from parsec_python.models import SpinSectorFallbackRecord

        problem = self.spin(1.0)
        fast = self._run(problem, {SPIN_FLOOR_SWITCH: "off", GUARD_SHORTCUT_SWITCH: "auto"})
        slow = self._run(problem, {SPIN_FLOOR_SWITCH: "off", GUARD_SHORTCUT_SWITCH: "off"})
        abelian = self._run(problem, {"PARSEC_ORBITAL_GROUP_SPIN": "off"})

        def notes(run):
            return [(record.route, record.abelian_skipped) for record in run.records
                    if isinstance(record, SpinSectorFallbackRecord)]

        self.assertEqual(notes(fast), [("weighted", True), ("spin", False)])
        self.assertEqual(notes(slow), [("weighted", False), ("spin", False)])
        self.assertEqual(notes(abelian), [("spin", False)])
        self.assertIsNone(fast.reduction)
        total = float(abelian.result.energies.total)
        self.assertEqual(float(fast.result.energies.total), total)
        self.assertEqual(float(slow.result.energies.total), total)
        self.assertEqual(fast.result.iterations, abelian.result.iterations)


if __name__ == "__main__":
    unittest.main()
