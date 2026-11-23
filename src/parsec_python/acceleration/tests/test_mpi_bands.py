"""The band k-points of a Gamma cell dealt to MPI ranks (``PARSEC_MPI_ACCELERATED_BANDS``).

Under ``--mpi`` every rank holds the converged potential of the accelerated SCF and
:func:`parsec_python.acceleration.SCF.band_sectors.solve_band_sectors` deals the chunks of k-points to the ranks, each
solved by the solver of one process, then gathers the levels on every rank.  The tests stand two ranks in for MPI (a
context whose ``allgather`` passes the parts of the ranks along), run rank 1 and then rank 0 on the silicon cube of
the band tests and check that rank 0 writes the bands of one process: with the same GPUs on both ranks, and with two
GPUs on one rank and one on the other (the ranks agree on one chunk length); and that a partition the ranks do not
agree on stops the run instead of writing levels no rank solved.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.tests.test_band_kpoint_sectors import PATH, SETTINGS, _cupy_ready
from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon


class _TwoRanks:
    """The communicator of rank ``rank`` of two, the ranks run one after the other.

    ``allgather`` of a chunk length returns ``chunks`` (the lengths of both ranks); of a part of the levels it keeps
    this rank's part and returns the parts known, a rank not run yet standing for the k-points this rank did not solve
    (zero levels), as its part would.
    """

    size = 2

    def __init__(self, rank: int, parts: dict, chunks, count: int) -> None:
        self.rank = rank
        self.comm = self
        self.parts = parts
        self.chunks = list(chunks)
        self.count = count

    def allgather(self, value):
        if isinstance(value, (int, np.integer)):
            return list(self.chunks)
        self.parts[self.rank] = value
        found, _ = value
        width = next(iter(found.values()))[0].shape
        rest = ({k: (np.zeros(width), 0.0) for k in range(self.count) if k not in found}, True)
        return [self.parts.get(rank, rest) for rank in range(self.size)]


def _bands(directory: Path, context=None, gpu_ids: str | None = None):
    import parsec_python.acceleration.cli as cli_module
    from parsec_python.acceleration.SCF import band_sectors, kpoint_sectors

    path = write_silicon(directory, PATH, positions=displaced_positions(0.0), settings=SETTINGS)
    variables = {"PARSEC_BAND_KPOINT_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_BANDS": "auto"}
    errors = StringIO()
    arguments = [str(path), "--quiet", "--backend", "cupy"] + ([] if gpu_ids is None else ["--gpu-ids", gpu_ids])
    with mock.patch.dict(os.environ, variables), mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0), \
            mock.patch.object(band_sectors, "_mpi_context", return_value=context), \
            redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main(arguments)
    bands = {}
    if status == 0:
        with np.load(directory / "parsec_python_bands.npz", allow_pickle=True) as data:
            bands = {key: data[key] for key in data.files}
    output = directory / "parsec.out"
    log = output.read_text(encoding="utf-8", errors="replace") if output.exists() else ""
    return status, bands, errors.getvalue() + log


def _device_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # pragma: no cover
        return 0


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class DealtBandTests(unittest.TestCase):
    def _two_ranks(self, root: Path, chunks_one, chunks_zero, ids_one=None, ids_zero=None):
        status, alone, errors = _bands(root / "one")
        self.assertEqual(status, 0, errors[-4000:])
        count = int(np.asarray(alone["eigenvalues_ry"]).shape[1])
        parts: dict = {}
        first = _bands(root / "rank1", _TwoRanks(1, parts, chunks_one, count), ids_one)
        second = _bands(root / "rank0", _TwoRanks(0, parts, chunks_zero, count), ids_zero)
        return alone, parts, first, second

    def _check(self, alone, parts, first, second) -> None:
        for status, _, errors in (first, second):
            self.assertEqual(status, 0, errors[-4000:])
        mine, theirs = parts[0][0], parts[1][0]
        self.assertTrue(mine and theirs)
        self.assertFalse(set(mine) & set(theirs))  # each k-point solved by one rank
        self.assertEqual(sorted(set(mine) | set(theirs)), list(range(np.asarray(alone["eigenvalues_ry"]).shape[1])))
        gathered = second[1]
        np.testing.assert_allclose(gathered["eigenvalues_ry"], alone["eigenvalues_ry"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(gathered["residual_norms"], alone["residual_norms"], rtol=0, atol=1e-12)

    def test_two_ranks_give_the_bands_of_one_process(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            self._check(*self._two_ranks(Path(name), (2, 2), (2, 2)))

    @unittest.skipUnless(_device_count() >= 3, "two ranks of two GPUs and one GPU need three GPUs")
    def test_ranks_of_different_gpu_counts_deal_one_partition(self) -> None:
        # rank 0 sees GPUs 0 and 1 (chunks of two k-points), rank 1 GPU 2 (chunks of one): both deal chunks of one
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            self._check(*self._two_ranks(Path(name), (2, 1), (2, 1), ids_one="2", ids_zero="0,1"))

    def test_a_partition_the_ranks_do_not_share_stops_the_run(self) -> None:
        # rank 1 deals chunks of one k-point, rank 0 chunks of three: some k-points are solved twice, others never
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            _, _, first, second = self._two_ranks(Path(name), (1, 1), (3, 3))
        self.assertEqual(first[0], 0, first[2][-4000:])
        self.assertNotEqual(second[0], 0)
        self.assertIn("did not solve every band k-point exactly once", second[2])


if __name__ == "__main__":
    unittest.main()
