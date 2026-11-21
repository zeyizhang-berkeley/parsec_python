"""Nested sharing: ranks of several GPUs each share a sector basis over all of them (CPU parts).

One rank per node (the default launch of ``scripts/run_multi_gpu_scf.sh``) held its share of a shared basis on its
first GPU (diamond 6^3 without symmetry: 327.0 s on two nodes against 36.8 s on one).  With
``PARSEC_MPI_NESTED_SHARING=auto`` (the default) and a CHEBFF first solve the ranks sharing that sector spread their
shares over all their GPUs (``Eigensolvers/mpi_nested_basis.py``); ``groups`` lets groups of such ranks share
several sectors too.  Here the switch, the decision, the placement and the
launch reading of the runner, on fake communicators; the GPU parts are in ``test_mpi_nested_basis``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from parsec_python.acceleration.experimental.mpi_scf import (
    MPISectorContext,
    nested_groups_apply,
    nested_sharing_applies,
    nested_sharing_setting,
    sector_device_groups,
)
from parsec_python.acceleration.tests.thread_mpi import run_ranks

_OFF = {"PARSEC_MPI_NESTED_SHARING": "off"}
_GROUPS = {"PARSEC_MPI_NESTED_SHARING": "groups"}


class _Comm:
    def __init__(self, rank, size):
        self.rank, self.size = rank, size


class SettingTests(unittest.TestCase):
    def test_values(self):
        for value, expected in (("", "auto"), (" AUTO ", "auto"), ("off", "off"), ("Groups", "groups")):
            with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": value}):
                self.assertEqual(nested_sharing_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "on"}), self.assertRaises(ValueError):
            nested_sharing_setting()

    def test_the_decision(self):
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto"}):
            self.assertTrue(nested_sharing_applies(2, 4, divided=True))
            # One rank, one GPU per rank, --rank-per-gpu, a hybrid's exchange, a CHEBDAV first solve: no.
            self.assertFalse(nested_sharing_applies(1, 4, divided=True))
            self.assertFalse(nested_sharing_applies(2, 1, divided=True))
            self.assertFalse(nested_sharing_applies(8, 4, gpu_per_rank=True, divided=True))
            self.assertFalse(nested_sharing_applies(2, 4, exchange=True, divided=True))
            self.assertFalse(nested_sharing_applies(2, 4, divided=False))
            # Groups over several sectors only with the value groups.
            self.assertFalse(nested_groups_apply(2, 4, divided=True))
        with patch.dict(os.environ, _GROUPS):
            self.assertTrue(nested_sharing_applies(2, 4, divided=True))
            self.assertTrue(nested_groups_apply(2, 4, divided=True))
            self.assertFalse(nested_groups_apply(8, 4, gpu_per_rank=True, divided=True))
            self.assertFalse(nested_groups_apply(2, 4, divided=False))
        with patch.dict(os.environ, _OFF):
            self.assertFalse(nested_sharing_applies(2, 4, divided=True))
            self.assertFalse(nested_groups_apply(2, 4, divided=True))
        # An unknown value is refused wherever the decision is asked.
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "yes"}), self.assertRaises(ValueError):
            nested_sharing_applies(2, 4, divided=True)


class PlacementTests(unittest.TestCase):
    def _context(self, rank, size, *, devices=4, divided=True):
        context = MPISectorContext(_Comm(rank, size))
        context.node_devices = devices
        context.divided_bases = divided
        return context

    def test_a_shared_sector_takes_every_gpu_of_a_rank(self):
        for rank in range(2):
            self.assertEqual(sector_device_groups(1, 2, rank, (0, 1, 2, 3), shared=True, nested=True),
                             {0: (0, 1, 2, 3)})
            # As before without nested sharing: the first GPU.
            self.assertEqual(sector_device_groups(1, 2, rank, (0, 1, 2, 3), shared=True), {0: (0,)})

    def test_the_context_shares_where_nested_sharing_applies(self):
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto", "PARSEC_SECTOR_PLACEMENT": "size"}):
            # The default: the one sector over every GPU of every rank, several sectors as before.
            context = self._context(0, 4)
            self.assertTrue(context.nested_sharing)
            self.assertFalse(context.nested_groups)
            self.assertTrue(context._shared(1))
            self.assertEqual(context.planned_groups(1, (0, 1, 2, 3)), {0: (0, 1, 2, 3)})
            # Two sectors on four nodes: no layout of one rank per node takes them, so they are shared in groups
            # (since 2026-11-20; a ValueError in the preparation before), but the driver is still told that they
            # do not divide, so that it keeps a group of before that does.
            self.assertTrue(context._shared(2))
            self.assertFalse(context._shared(2, last_resort=False))
            self.assertFalse(context.divides(2))
            self.assertEqual(set(context.planned_groups(2, (0, 1, 2, 3)).values()), {(0, 1, 2, 3)})
            self.assertFalse(context._shared(6))
            # One rank per sector group, the root the lightest (one sector over its GPUs), as before.
            self.assertEqual(context.planned_groups(6, (0, 1, 2, 3)), {2: (0, 1, 2, 3)})
        with patch.dict(os.environ, {**_GROUPS, "PARSEC_SECTOR_PLACEMENT": "size"}):
            context = self._context(0, 4)
            self.assertTrue(context.nested_sharing)
            self.assertTrue(context.nested_groups)
            self.assertTrue(context._shared(1))
            # Two sectors on four nodes: refused by one rank per node before, shared in pairs now.
            self.assertTrue(context._shared(2))
            self.assertTrue(context.divides(2))
            # Six equal sectors on four ranks: two groups of two (1.5 sectors per rank) beat 2/2/1/1.
            self.assertTrue(context._shared(6))
            self.assertEqual(context.planned_groups(6, (0, 1, 2, 3)), {0: (0, 1, 2, 3), 2: (0, 1, 2, 3),
                                                                      4: (0, 1, 2, 3)})
            # Four sectors on four ranks: one rank each, as before (nothing shared).
            self.assertFalse(context._shared(4))
            # After a CHEBDAV first solve nothing changes.
            chebdav = self._context(0, 4, divided=False)
            self.assertFalse(chebdav.nested_sharing)
            self.assertFalse(chebdav.divides(2))
            self.assertEqual(chebdav.planned_groups(1, (0, 1, 2, 3)), {0: (0,)})
            # Fewer sectors than ranks with no layout: a message that names what runs them.
            with self.assertRaisesRegex(ValueError, "fewer than the 4 MPI ranks.*name Eigensolver: chebff.*--rank-per-gpu"):
                chebdav.planned_groups(2, (0, 1, 2, 3))
        with patch.dict(os.environ, {**_OFF, "PARSEC_SECTOR_PLACEMENT": "size"}):
            context = self._context(0, 4)
            self.assertFalse(context.nested_sharing)
            self.assertFalse(context.divides(2))
            self.assertFalse(context._shared(2))
            self.assertFalse(context._shared(6))
            self.assertEqual(context.planned_groups(1, (0, 1, 2, 3)), {0: (0,)})
            with self.assertRaisesRegex(ValueError, "leave PARSEC_MPI_NESTED_SHARING at auto.*at most 2 node"):
                context.planned_groups(2, (0, 1, 2, 3))
            # One GPU per rank without --rank-per-gpu: no nested sharing to name.
            single = self._context(0, 4, devices=1)
            message = single._fewer_sectors_refusal(2, False)
            self.assertNotIn("PARSEC_MPI_NESTED_SHARING", message)
            self.assertIn("--rank-per-gpu", message)
        with patch.dict(os.environ, {**_GROUPS, "PARSEC_SECTOR_PLACEMENT": "cyclic"}):
            # Placed by count, more ranks than several sectors divide with one GPU per rank alone, but the
            # sectors that only groups can take are shared in groups of consecutive ranks.
            self.assertFalse(self._context(0, 4).divides(2))
            self.assertEqual(set(self._context(0, 4).planned_groups(2, (0, 1, 2, 3)).values()), {(0, 1, 2, 3)})
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto", "PARSEC_SECTOR_PLACEMENT": "size"}):
            # A hybrid's ranks may own no sector, and --rank-per-gpu shares them: no message there.
            hybrid = self._context(0, 4)
            hybrid.exchange = True
            self.assertIsNone(hybrid._fewer_sectors_refusal(2, False))
            self.assertIsNone(self._context(0, 4)._fewer_sectors_refusal(4, False))
            self.assertIsNone(self._context(0, 4)._fewer_sectors_refusal(2, True))

    def test_fewer_sectors_than_nodes_are_shared_under_auto(self):
        def configure(ctx):
            context = MPISectorContext(ctx.comm)
            context.node_devices = 4
            context.divided_bases = True
            try:
                groups = context.configure(2, (0, 1, 2, 3), costs=[182718.0, 178378.0])
            except Exception as error:  # noqa: BLE001
                return type(error).__name__, str(error)
            return context.shared, context.nested, groups, context.rank_groups

        for switch in ({"PARSEC_MPI_NESTED_SHARING": "auto"}, _GROUPS):
            with patch.dict(os.environ, {**switch, "PARSEC_SECTOR_PLACEMENT": "size"}):
                results = run_ranks(3, configure)
            self.assertEqual(len({result[3] for result in results}), 1)
            for shared, nested, groups, rank_groups in results:
                self.assertTrue(shared and nested)
                self.assertEqual(set(groups.values()), {(0, 1, 2, 3)})
                # Every rank solves a sector.
                self.assertEqual(sorted({rank for ranks in rank_groups for rank in ranks}), [0, 1, 2])
            # Both values agree: auto takes the layout that groups takes there.
            if switch is _GROUPS:
                self.assertEqual(results, auto)
            auto = results
        with patch.dict(os.environ, {**_OFF, "PARSEC_SECTOR_PLACEMENT": "size"}):
            results = run_ranks(3, configure)
        for name, message in results:
            self.assertEqual(name, "MPISCFError")
            self.assertIn("fewer than the 3 MPI ranks", message)

    def test_configure_marks_the_nested_layout_on_every_rank(self):
        def configure(ctx):
            context = MPISectorContext(ctx.comm)
            context.node_devices = 4
            context.divided_bases = True
            groups = context.configure(6, (0, 1, 2, 3))
            return (context.shared, context.nested, groups, context.rank_groups,
                    tuple(context.sector_comm.allgather(ctx.rank)))

        with patch.dict(os.environ, {**_GROUPS, "PARSEC_SECTOR_PLACEMENT": "size"}):
            results = run_ranks(4, configure)
        plans = {result[3] for result in results}
        self.assertEqual(len(plans), 1)
        plan = plans.pop()
        self.assertEqual(sorted({ranks for ranks in plan}), [(0, 1), (2, 3)])
        for rank, (shared, nested, groups, _plan, members) in enumerate(results):
            self.assertTrue(shared and nested)
            self.assertEqual(set(groups.values()), {(0, 1, 2, 3)})
            self.assertEqual(len(groups), 3)
            self.assertEqual(members, (0, 1) if rank < 2 else (2, 3))
        for switch in (_OFF, {"PARSEC_MPI_NESTED_SHARING": "auto"}):
            with patch.dict(os.environ, {**switch, "PARSEC_SECTOR_PLACEMENT": "size"}):
                results = run_ranks(4, configure)
            for shared, nested, groups, _plan, _members in results:
                self.assertFalse(shared or nested)


class HintTests(unittest.TestCase):
    def test_the_note_names_groups_where_a_margin_was_cut(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _nested_groups_note

        def configure(ctx):
            context = MPISectorContext(ctx.comm)
            context.node_devices = 4
            context.divided_bases = True
            # Ten nearly equal sectors on four nodes (diamond 22^3): 3/3/2/2 per rank, two groups of two under groups.
            context.configure(10, (0, 1, 2, 3), costs=[1.0 + 0.001 * sector for sector in range(10)])
            return context.nested_groups_hint(), context

        cut = SimpleNamespace(sector_margin_record={"0": {"fraction": 0.5}, "1": {"fraction": 1.0}})
        full = SimpleNamespace(sector_margin_record={"0": {"fraction": 1.0}, "1": {"fraction": 1.0}})
        with patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto", "PARSEC_SECTOR_PLACEMENT": "size"}):
            results = run_ranks(4, configure)
            self.assertEqual({hint for hint, _context in results}, {True})
            context = results[0][1]
            note = _nested_groups_note(context, cut)
            self.assertIn("PARSEC_MPI_NESTED_SHARING=groups", note)
            self.assertIn("1 of the 2 sectors", note)
            # The three ways with the measured cost of each, the fastest first (review of 2026-11-20).
            self.assertLess(note.index("PARSEC_CUPY_SECTOR_MARGIN=fixed"), note.index("--rank-per-gpu"))
            self.assertLess(note.index("--rank-per-gpu"), note.index("PARSEC_MPI_NESTED_SHARING=groups"))
            for figure in ("1044 s", "264.2 s", "322.9 s", "518.6 s"):
                self.assertIn(figure, note)
            self.assertIsNone(_nested_groups_note(context, full))
            self.assertIsNone(_nested_groups_note(context, SimpleNamespace()))
            self.assertIsNone(_nested_groups_note(None, cut))
            # Four sectors on four nodes: one each, no group would form.
            four = MPISectorContext(_Comm(0, 4))
            four.node_devices, four.divided_bases = 4, True
            four.representation_count, four.sector_costs = 4, (1.0, 1.0, 1.0, 1.0)
            self.assertFalse(four.nested_groups_hint())
            # --rank-per-gpu, a CHEBDAV first solve, one GPU per rank: no hint.
            for change in ({"gpu_per_rank": True}, {"divided_bases": False}, {"node_devices": 1}):
                other = MPISectorContext(_Comm(0, 4))
                other.node_devices, other.divided_bases = 4, True
                other.representation_count, other.sector_costs = 10, (1.0,) * 10
                self.assertTrue(other.nested_groups_hint())
                for key, value in change.items():
                    setattr(other, key, value)
                self.assertFalse(other.nested_groups_hint())
        with patch.dict(os.environ, {**_GROUPS, "PARSEC_SECTOR_PLACEMENT": "size"}):
            # Already in groups: nothing to name.
            for hint, context in run_ranks(4, configure):
                self.assertFalse(hint)
                self.assertTrue(context.shared)
        with patch.dict(os.environ, {**_OFF, "PARSEC_SECTOR_PLACEMENT": "size"}):
            self.assertEqual({hint for hint, _context in run_ranks(4, configure)}, {False})

    def test_the_line_of_a_nested_layout(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _nested_sharing_line

        def configure(ctx, representations):
            context = MPISectorContext(ctx.comm)
            context.node_devices = 4
            context.divided_bases = True
            context.configure(representations, (0, 1, 2, 3))
            return context

        environment = {"PARSEC_MPI_NESTED_SHARING": "auto", "PARSEC_SECTOR_PLACEMENT": "size",
                       "PARSEC_MPI_SHARED_TRANSPORT": "cuda"}
        with patch.dict(os.environ, environment):
            # The one sector shared by both ranks, over the four GPUs of each.
            context = run_ranks(2, lambda ctx: configure(ctx, 1))[0]
            line = _nested_sharing_line(context, 4)
            self.assertIn("PARSEC_MPI_NESTED_SHARING=auto", line)
            self.assertIn("1 sector(s) shared by groups of 2 rank(s)", line)
            self.assertIn("over its 4 GPUs", line)
            self.assertIn("device arrays", line)
            # Several sectors, one rank each: nothing shared, no line (the launches of before print as before).
            self.assertIsNone(_nested_sharing_line(run_ranks(2, lambda ctx: configure(ctx, 6))[0], 4))
            self.assertIsNone(_nested_sharing_line(None, 4))
        with patch.dict(os.environ, {**environment, "PARSEC_MPI_SHARED_TRANSPORT": "host"}):
            self.assertIn("host-staged arrays", _nested_sharing_line(run_ranks(2, lambda ctx: configure(ctx, 1))[0], 4))
        with patch.dict(os.environ, {**environment, **_OFF, "PARSEC_SINGLE_SECTOR_NODE_LAYOUT": "allow"}):
            # Shared on the first GPU of each rank, as before: no line (that layout is
            # refused under off unless PARSEC_SINGLE_SECTOR_NODE_LAYOUT=allow, test_runner_fixes.NestedLayoutTests).
            self.assertIsNone(_nested_sharing_line(run_ranks(2, lambda ctx: configure(ctx, 1))[0], 4))

    def test_the_line_names_the_groups_attached(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _nested_attachment, _nested_sharing_line

        def attach(ctx, attached):
            context = MPISectorContext(ctx.comm)
            context.node_devices = 4
            context.divided_bases = True
            context.configure(2, (0, 1, 2, 3), costs=[2.0, 1.0])
            solver = SimpleNamespace(nested_representations=tuple(sector for sector in context.owned_sectors
                                                                   if sector in attached))
            return context, _nested_attachment(ctx.comm, context, solver)

        environment = {"PARSEC_MPI_NESTED_SHARING": "auto", "PARSEC_SECTOR_PLACEMENT": "size",
                       "PARSEC_MPI_SHARED_TRANSPORT": "cuda"}
        with patch.dict(os.environ, environment):
            results = run_ranks(3, lambda ctx: attach(ctx, (0, 1)))
            self.assertEqual({attachment for _context, attachment in results}, {(0, 1)})
            context, attachment = results[0]
            line = _nested_sharing_line(context, 4, attachment)
            # Costs 2:1 on three nodes: the costlier sector on two of them, the other on the third.
            self.assertIn("2 sector(s) shared by groups of 1/2 rank(s)", line)
            # The line counts what the eigensolvers attached, not what the context planned.
            context, attachment = run_ranks(3, lambda ctx: attach(ctx, (1,)))[0]
            self.assertEqual(attachment, (1,))
            self.assertIn("1 sector(s) shared", _nested_sharing_line(context, 4, attachment))
            self.assertIsNone(_nested_sharing_line(context, 4, ()))
            # Nothing shared over the GPUs: no collective, no line.
            self.assertIsNone(_nested_attachment(None, None, None))
        with patch.dict(os.environ, {**environment, **_OFF}):
            self.assertIsNone(_nested_attachment(_Comm(0, 1), MPISectorContext(_Comm(0, 4)), SimpleNamespace()))


class LaunchTests(unittest.TestCase):
    def _input(self, folder, *lines):
        path = Path(folder) / "parsec.in"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_the_runner_reads_the_launch_before_mpi(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _input_first_solve_is_chebff, _nested_launch

        with tempfile.TemporaryDirectory() as folder:
            chebff = self._input(folder, "Eigensolver: chebff")
            self.assertTrue(_input_first_solve_is_chebff(chebff))
            plain = self._input(folder, "Grid_Spacing: 0.5")
            self.assertFalse(_input_first_solve_is_chebff(plain))
            chebff = self._input(folder, "Eigensolver: chebff")
            args = SimpleNamespace(serial_control=False, devices=(0, 1, 2, 3), input=chebff)
            slurm = {"SLURM_NTASKS": "2", "SLURM_NTASKS_PER_NODE": "1"}
            with patch.dict(os.environ, {**slurm, "PARSEC_MPI_NESTED_SHARING": "auto"}):
                self.assertTrue(_nested_launch(args))
                self.assertFalse(_nested_launch(SimpleNamespace(**{**vars(args), "devices": (0,)})))
                self.assertFalse(_nested_launch(SimpleNamespace(**{**vars(args), "serial_control": True})))
            with patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_NTASKS_PER_NODE": "4(x2)"}):
                self.assertFalse(_nested_launch(args))
            with patch.dict(os.environ, {**slurm, **_OFF}):
                self.assertFalse(_nested_launch(args))
            with patch.dict(os.environ, {"SLURM_NTASKS": "1", "SLURM_NTASKS_PER_NODE": "1"}):
                self.assertFalse(_nested_launch(args))

    def test_gpu_support_only_where_the_transport_layer_is_loaded(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _gpu_transport_layer_loaded, _nested_gpu_support

        with tempfile.TemporaryDirectory() as folder:
            linked = Path(folder) / "linked"
            linked.write_text("7f0000000000-7f0000001000 r-xp 00000000 00:00 1 /opt/cray/pe/mpich/9.1.0/gtl/lib/"
                              "libmpi_gtl_cuda.so.0.1.0\n7f0000002000-7f0000003000 r-xp 00000000 00:00 2 /usr/lib/libc.so\n")
            plain = Path(folder) / "plain"
            plain.write_text("7f0000002000-7f0000003000 r-xp 00000000 00:00 2 /opt/cray/pe/lib64/libmpi_gnu.so.12\n")
            self.assertTrue(_gpu_transport_layer_loaded(linked))
            self.assertFalse(_gpu_transport_layer_loaded(plain))
            self.assertFalse(_gpu_transport_layer_loaded(Path(folder) / "missing"))
        # A nested launch asks for GPU support where the layer is loaded, and is host-staged with a reason otherwise
        # (Cray MPICH stops in MPI_Init when the flag asks for a layer that is not linked).
        self.assertEqual(_nested_gpu_support(True, "auto", True), ("1", None))
        flag, note = _nested_gpu_support(True, "auto", False)
        self.assertIsNone(flag)
        self.assertIn("libmpi_gtl_cuda", note)
        # PARSEC_MPI_SHARED_TRANSPORT=cuda asks for it anyway, host never; no nested launch changes nothing.
        self.assertEqual(_nested_gpu_support(True, "CUDA", False), ("1", None))
        self.assertEqual(_nested_gpu_support(True, "host", True), (None, None))
        self.assertEqual(_nested_gpu_support(False, "auto", True), (None, None))


if __name__ == "__main__":
    unittest.main()
