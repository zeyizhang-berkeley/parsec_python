"""The start of a symmetry sector that grows during the SCF: its saved vectors and new random columns.

The spectral bracket (``symmetry.CuPySymmetrySCFEigensolver._ensure_spectral_bracket``) grows a sector that misses
states.  Its new states need a first solve, and a first solve from random vectors throws away the converged vectors
of the sector, so that its occupied levels restart unconverged and the SCF jumps.  With
``PARSEC_CUPY_SECTOR_WARM_GROWTH`` a CHEBFF first solve starts instead from the saved vectors, where a later pass
has refined them, followed by as many columns of the usual random stream as the sector gains.

The saved columns wait on the host while the new basis is formed (:func:`saved_columns`), so that a device holds no
more than it holds for a first solve from random vectors.  They are then written into the leading columns of the
new basis: a single array here (:func:`place_saved_columns`), the blocks of a basis that devices or MPI ranks share
in ``fill_leading`` of their groups.  A process stages them only where the host has room for them
(:func:`host_room`), and the MPI ranks that share a basis start from it only where every one of them staged its
columns (:func:`staged_on_every_rank`); otherwise the sector grows from random vectors, as before.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from ..backends.cupy import require_cupy
from .spill_buffers import copies_directly, host_available_bytes

_VALUES = {"auto", "off"}
# The share of the available host memory that the saved columns of a process may take (see host_room).
_HOST_SHARE = 0.5
_reserved_lock = threading.Lock()
_reserved_bytes = 0


def sector_warm_growth_setting() -> str:
    """``PARSEC_CUPY_SECTOR_WARM_GROWTH``: ``auto`` (the default) or ``off``.

    ``auto`` starts the CHEBFF first solve of a sector that grows in the spectral bracket from its saved vectors
    and new random columns, where a later pass has refined those vectors.  The vectors of the first solve of a run,
    which no later pass refined, are not kept.  A CHEBDAV first solve (the input default) grows as before: it draws
    a new trial block only when fewer than a block of active vectors remain, so it would read only the first six
    saved columns, and from those it ran out of its outer restarts in metals (fcc Al of 108 atoms: 14 -> 25 SCF
    steps).  ARPACK has no start block.  ``off`` starts every growth from random vectors, as before.
    """

    value = os.environ.get("PARSEC_CUPY_SECTOR_WARM_GROWTH", "auto").strip().lower() or "auto"
    if value not in _VALUES:
        raise ValueError("PARSEC_CUPY_SECTOR_WARM_GROWTH must be auto or off")
    return value


def new_column_scale(rows: int) -> float:
    """The factor that gives a column of the random start (uniform in [-1, 1]) about unit norm, as the saved have."""

    return 1.0 / max(np.sqrt(int(rows) / 3.0), 1.0)


@dataclass
class SavedColumns:
    """The leading ``kept`` columns of a saved basis, on the host.

    ``layout`` is ``None`` where ``arrays[None]`` holds all of them.  Otherwise it is the layout of those columns
    over the ranks (virtual ranks of a nested group), and ``arrays`` maps each rank of this process to its columns,
    side by side in the order of the layout.
    """

    kept: int
    layout: Any
    arrays: dict


def _download(cp: Any, block: Any, out: np.ndarray | None = None) -> np.ndarray:
    """``block`` of the current device in a Fortran-ordered host array, by one copy.

    ``ndarray.get(out=)`` takes no second array on the device
    (``spill_buffers``); ``cupy.asnumpy`` in C order takes one of the size of the block.
    """

    if out is None:
        out = np.empty(block.shape, dtype=np.float64, order="F")
    if copies_directly(block):
        block.get(out=out)
        cp.cuda.get_current_stream().synchronize()
    elif block.size:
        out[...] = cp.asnumpy(block, order="A")
    return out


def saved_columns(vectors: Any, keep: int) -> SavedColumns:
    """The leading ``keep`` columns of a saved basis (an array or a shared basis) on the host.

    Each device writes the columns it holds, at the same time as the others of its group.
    """

    from .distributed_state import DistributedBasis
    from .mpi_nested_basis import NestedBasis
    from .mpi_shared_basis import SharedBasis

    cp, _ = require_cupy()
    keep = int(keep)
    if isinstance(vectors, NestedBasis):
        held = vectors[:, :keep]

        def take_nested(virtual, _device):
            return virtual, _download(cp, held.blocks[virtual])

        return SavedColumns(keep, held.layout, dict(held.group.map(take_nested)))
    if isinstance(vectors, SharedBasis):
        held = vectors[:, :keep]
        block = held.local
        cp.cuda.Device(int(block.device.id)).synchronize()
        with cp.cuda.Device(int(block.device.id)):
            array = _download(cp, block)
        return SavedColumns(keep, held.layout, {held.group.rank: array})
    if isinstance(vectors, DistributedBasis):
        held = vectors[:, :keep]
        whole = np.empty((int(held.shape[0]), keep), dtype=np.float64, order="F")

        def take(index, _device):
            block = held.blocks[index]
            for column, offset, count in held.layout.spans(index):
                _download(cp, block[:, offset : offset + count], whole[:, column : column + count])

        held.group.map(take)
        return SavedColumns(keep, None, {None: whole})
    if isinstance(vectors, np.ndarray):
        # A basis spilled to the host: its buffer is written again only behind the solve that reads this.
        return SavedColumns(keep, None, {None: vectors[:, :keep]})
    cp.cuda.get_current_stream().synchronize()
    return SavedColumns(keep, None, {None: _download(cp, vectors[:, :keep])})


def host_bytes(vectors: Any, keep: int) -> int:
    """The host bytes that :func:`saved_columns` takes for the leading ``keep`` columns of ``vectors``.

    A basis spilled to the host is read where it lies; a shared basis puts the columns of this process on the host,
    a basis that the devices of one process share all of them.
    """

    if isinstance(vectors, np.ndarray):
        return 0
    return int(vectors[:, : int(keep)].nbytes)


@contextmanager
def host_room(nbytes: int) -> Iterator[bool]:
    """Whether ``nbytes`` fit on the host now; while they are being staged they are counted as taken.

    They fit where they are at most half of the memory that the host reports as available
    (``spill_buffers.host_available_bytes``), less what other sectors of this process are staging at the same
    time; and where the host reports nothing.
    """

    global _reserved_bytes
    nbytes = int(nbytes)
    with _reserved_lock:
        available = host_available_bytes() if nbytes else None
        fits = available is None or nbytes <= _HOST_SHARE * (available - _reserved_bytes)
        if fits:
            _reserved_bytes += nbytes
    try:
        yield fits
    finally:
        if fits:
            with _reserved_lock:
                _reserved_bytes -= nbytes


def staged_on_every_rank(vectors: Any, staged: bool) -> bool:
    """Whether every MPI rank that shares ``vectors`` staged its saved columns (``staged`` on this rank).

    The ranks of a basis that they share (``mpi_shared_basis.SharedBasis`` and its nested form) take the same start
    or all would wait in different collectives: they agree by one ``allreduce`` on the communicator of the basis.
    Every rank of the group calls this for the same sector in the same order.  A basis of one process returns
    ``staged``.
    """

    from .mpi_shared_basis import SharedBasis

    if not isinstance(vectors, SharedBasis):
        return bool(staged)
    comm = vectors.group.comm
    return int(comm.allreduce(0 if staged else 1)) == 0


def place_saved_columns(vectors: Any, saved: SavedColumns) -> None:
    """Write the saved columns into the leading columns of ``vectors``, and scale the columns behind them."""

    rows, columns = map(int, vectors.shape)
    kept = min(int(saved.kept), columns)
    vectors[:, :kept].set(np.asfortranarray(saved.arrays[None][:, :kept]))
    if kept < columns:
        vectors[:, kept:] *= new_column_scale(rows)


__all__ = [
    "SavedColumns",
    "host_bytes",
    "host_room",
    "new_column_scale",
    "place_saved_columns",
    "saved_columns",
    "sector_warm_growth_setting",
    "staged_on_every_rank",
]
