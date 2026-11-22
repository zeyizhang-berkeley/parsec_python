"""A sector basis shared by MPI ranks that each spread their share over the GPUs of their node.

:mod:`mpi_shared_basis` shares the ``N x m`` basis of a sector among the
ranks of a communicator, one GPU each.  With one rank per node (the default
launch of ``scripts/run_multi_gpu_scf.sh``) a rank drives the four GPUs of
its node, and before 2026-11-19 it held its share of a shared basis on the
first of them: diamond 6^3 without symmetry took 327.0 s on two nodes against
36.8 s on one, GPUs 1-3 idle and the exchanges staged through the host.  Here
the ``R`` ranks of a group with ``D`` devices each act as ``R * D`` virtual
ranks, rank ``r``'s device ``i`` being virtual rank ``r * D + i``:

* every device holds the columns of its virtual rank (all grid rows), with a
  replica of the sector operator (:class:`distributed_filter.DistributedFilter`),
  and filters them without communication;
* the Rayleigh--Ritz step is that of :class:`mpi_shared_basis.MPIShardGroup`
  on ``R * D`` virtual ranks, run by one :class:`_VirtualShardGroup` per
  device on a thread of its own: the same routes (``block`` in the runner),
  rounds, Gram products, small solve and helpers;
* their collectives go to a :class:`_NodeHub`, which carries each out on the
  calling (main) thread of the rank, the only one that calls MPI (the runner
  initializes MPI with ``THREAD_FUNNELED``), once every device thread of the
  rank has posted it: the pieces of an ``Alltoallv`` between two devices of
  the rank as device-to-device copies, those to and from other ranks as
  ``Isend``/``Irecv`` of device buffers (GPU-aware MPI, ``MPICH_GPU_SUPPORT_ENABLED=1``)
  or of host arrays (``PARSEC_MPI_SHARED_TRANSPORT=host``) on a duplicate of
  the group's communicator; an ``Allreduce`` sums the devices of the rank in
  their order and then the ranks; ``Bcast``/``bcast`` reach one device of
  every rank through MPI and the others by copies;
* point-to-point messages stay inside a rank (the condition and eigh helpers
  of the small solve are taken among the devices of the root's rank, which
  holds three or more of them on an A100 node; with two the root solves the
  eigenproblem itself, as a group of two ranks does).

The density and the nonlocal force terms of the sector are summed over the
devices of the rank (:meth:`NestedBasis.density`, :meth:`NestedShardGroup.map`)
and then over the ranks by the commands of :class:`experimental.mpi_scf.MPISectorContext`.
``PARSEC_MPI_NESTED_SHARING`` (:func:`experimental.mpi_scf.nested_sharing_setting`)
selects the route; ``off`` keeps each rank's share on its first GPU, as
before.  Only a CHEBFF first solve divides a basis; another one leaves the
layout of before.
"""

from __future__ import annotations

import copy
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from time import perf_counter
from typing import Any
from weakref import ref

import numpy as np

from ..backends.cupy import require_cupy
from .chebyshev import FilterBlock
from .distributed_filter import DistributedFilter, starting_sigmas
from .distributed_state import SectorDeviceGroup, _as_layout, _peer_access
from .mpi_shared_basis import (
    MPIShardGroup,
    SharedBasis,
    _ends_the_job_on_failure,
    _skip,
    _stability_error,
    shared_transport,
)
from .rayleigh_ritz import GeneralizedRitzStabilityError


class _HubFailure(RuntimeError):
    """Another device thread of the rank failed: this one leaves the collective or message it waited in."""


class _Shim:
    """What :class:`mpi_shared_basis.MPIShardGroup` reads of the MPI module (``IN_PLACE``, ``DOUBLE``)."""

    IN_PLACE = object()
    DOUBLE = None


def _is_device(array) -> bool:
    return hasattr(array, "__cuda_array_interface__")


def _flat(array):
    """A contiguous array as one dimension (a view)."""

    return array.reshape(-1)


class _NodeHub:
    """The collectives of the device threads of one rank, carried out on the thread that calls :meth:`run`.

    ``comm`` is the real communicator of the group (one rank per node), ``devices`` the GPUs of this rank.
    Virtual rank ``rank * len(devices) + i`` is device ``i`` of rank ``rank``.  A device thread posts a
    collective with :meth:`call` and waits; once every device thread of the rank has posted its next one
    (the same kind, in the same order on every thread, as MPI asks of ranks), the thread in :meth:`run`
    carries it out with MPI and hands back the results.  An error on any device thread, or a thread that
    finishes while the others wait in a collective, ends the waits of the others with :class:`_HubFailure`.
    """

    def __init__(self, comm, devices, transport: str) -> None:
        cp, _ = require_cupy()
        self.comm = comm
        self.ranks, self.rank = int(comm.size), int(comm.rank)
        self.devices = tuple(int(device) for device in devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("a nested group needs distinct devices")
        self.local = len(self.devices)
        self.size = self.ranks * self.local
        self.base = self.rank * self.local
        if transport not in {"cuda", "host"}:
            raise ValueError("transport must be cuda or host")
        self.transport = transport
        try:
            from mpi4py import MPI

            self._mpi = MPI if type(comm).__module__.startswith("mpi4py") else None
        except ImportError:
            self._mpi = None
        # The pieces of the exchanges travel between ranks on a communicator of their own.
        self._messages = comm.Dup() if self.ranks > 1 and hasattr(comm, "Dup") else comm
        self._condition = threading.Condition()
        self._posted: dict[int, tuple] = {}
        self._results: dict[int, Any] = {}
        self._sequence = [0] * self.local
        self._failure: str | None = None
        self._mail: dict[tuple[int, int, int], list] = {}
        self._running = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.local, thread_name_prefix="parsec-nested-device")
        self.streams = {}
        for device in self.devices:
            with cp.cuda.Device(device):
                self.streams[device] = cp.cuda.Stream(non_blocking=True)
            for peer in self.devices:
                _peer_access(cp, device, peer)
        # Wall seconds of the collectives on the calling thread, and the bytes this rank sent to other ranks.
        self.seconds = dict.fromkeys(("bcast", "Bcast", "Allreduce", "Alltoallv"), 0.0)
        self.remote_bytes = 0
        self.calls = 0

    # ----- device threads ----------------------------------------------------------------------------------

    def _check(self) -> None:
        if self._failure is not None:
            raise _HubFailure(self._failure)

    def call(self, index: int, kind: str, payload):
        """Post collective ``kind`` of device ``index`` and wait for its result (device threads only)."""

        with self._condition:
            self._check()
            sequence = self._sequence[index]
            self._sequence[index] += 1
            self._posted[index] = (sequence, kind, payload)
            self._condition.notify_all()
            while index not in self._results:
                self._check()
                self._condition.wait()
            return self._results.pop(index)

    def send(self, index: int, kind: str, payload, dest: int, tag: int) -> None:
        """A message of device ``index`` to virtual rank ``dest`` of this rank: an object at once, a buffer
        once the receiver has copied it (as a large ``MPI_Send``)."""

        owner, _ = divmod(int(dest), self.local)
        if owner != self.rank:
            raise NotImplementedError("a nested group sends point-to-point messages only between the devices of a rank")
        entry = {"kind": kind, "payload": payload, "done": False}
        with self._condition:
            self._check()
            self._mail.setdefault((self.base + index, int(dest), int(tag)), []).append(entry)
            self._condition.notify_all()
            if kind == "buffer":
                while not entry["done"]:
                    self._check()
                    self._condition.wait()

    def receive(self, index: int, kind: str, buffer, source: int, tag: int):
        """The next message from virtual rank ``source`` of this rank with ``tag``, to device ``index``."""

        owner, _ = divmod(int(source), self.local)
        if owner != self.rank:
            raise NotImplementedError("a nested group receives point-to-point messages only between the devices of a rank")
        key = (int(source), self.base + index, int(tag))
        with self._condition:
            while True:
                self._check()
                queue = self._mail.get(key)
                if queue:
                    entry = queue.pop(0)
                    break
                self._condition.wait()
        if entry["kind"] != kind:
            raise RuntimeError(f"virtual rank {self.base + index} expected a {kind} from {source}, not a {entry['kind']}")
        if kind == "object":
            return pickle.loads(entry["payload"])
        try:
            source_buffer = entry["payload"]
            if source_buffer.nbytes > buffer.nbytes:
                raise ValueError("a received message is longer than its buffer")
            if _is_device(buffer):
                cp, _ = require_cupy()
                stream = cp.cuda.get_current_stream()
                _flat(buffer)[: source_buffer.size].data.copy_from_device_async(
                    _flat(source_buffer).data, source_buffer.nbytes, stream)
                stream.synchronize()
            else:
                _flat(buffer)[: source_buffer.size] = _flat(source_buffer)
        finally:
            with self._condition:
                entry["done"] = True
                self._condition.notify_all()
        return None

    # ----- the calling thread ----------------------------------------------------------------------------------

    def run(self, jobs) -> list:
        """Run ``jobs[i]()`` on a thread of its own for device ``i`` and serve their collectives here.

        Returns one ``(True, value)`` or ``(False, exception)`` per device, in device order.  An error of the
        collectives themselves is raised after every device thread has stopped.
        """

        if len(jobs) != self.local:
            raise ValueError("one job per device of the rank")
        if not self._running.acquire(blocking=False):
            raise RuntimeError("a nested group runs one step at a time, on one thread")
        try:
            with self._condition:
                # A step starts afresh: a failed one may have left the threads at different collectives.
                self._failure = None
                self._posted.clear()
                self._results.clear()
                self._mail.clear()
                self._sequence = [0] * self.local
            futures = [self._executor.submit(self._job, index, job) for index, job in enumerate(jobs)]
            for future in futures:
                # Called once the future is done (or at once): the loop below then sees it done.
                future.add_done_callback(self._wake)
            error = None
            try:
                while True:
                    with self._condition:
                        while True:
                            done = sum(future.done() for future in futures)
                            if done == self.local or self._failure is not None:
                                posted = None
                                break
                            if len(self._posted) == self.local:
                                posted = dict(self._posted)
                                self._posted.clear()
                                break
                            if done and len(self._posted) + done >= self.local:
                                self._fail("a device thread of the rank finished while the others waited in a "
                                           "collective")
                                posted = None
                                break
                            self._condition.wait()
                    if posted is None:
                        break
                    results = self._serve(posted)
                    with self._condition:
                        self._results.update(results)
                        self._condition.notify_all()
            except BaseException as exc:
                error = exc
                self._fail(f"{type(exc).__name__}: {exc}")
            finally:
                wait(futures)
            if error is not None:
                raise error
            outcomes = []
            for future in futures:
                failure = future.exception()
                outcomes.append((True, future.result()) if failure is None else (False, failure))
            return outcomes
        finally:
            self._running.release()

    def _fail(self, message: str) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = message
            self._condition.notify_all()

    def _wake(self, _future=None) -> None:
        with self._condition:
            self._condition.notify_all()

    def _job(self, index: int, job):
        try:
            return job()
        except BaseException as exc:
            if not isinstance(exc, (GeneralizedRitzStabilityError, _HubFailure)):
                self._fail(f"device {self.devices[index]}: {type(exc).__name__}: {exc}")
            raise

    def _serve(self, posted: dict) -> dict:
        kinds = {(item[0], item[1]) for item in posted.values()}
        if len(kinds) != 1:
            raise RuntimeError(f"the devices of rank {self.rank} called different collectives: {sorted(kinds)}")
        _sequence, kind = kinds.pop()
        payloads = [posted[index][2] for index in range(self.local)]
        started = perf_counter()
        handler = {"bcast": self._bcast, "Bcast": self._Bcast, "Allreduce": self._Allreduce,
                   "Alltoallv": self._Alltoallv}.get(kind)
        if handler is None:
            raise RuntimeError(f"a nested group has no collective {kind}")
        values = handler(payloads)
        self.seconds[kind] += perf_counter() - started
        self.calls += 1
        return dict(enumerate(values))

    # ----- the collectives ---------------------------------------------------------------------------------------

    def _root_of(self, payloads, position: int) -> tuple[int, int]:
        roots = {int(payload[position]) for payload in payloads}
        if len(roots) != 1:
            raise RuntimeError("the devices of a rank named different roots")
        root = roots.pop()
        if not 0 <= root < self.size:
            raise ValueError(f"root {root} is no virtual rank of the group")
        return divmod(root, self.local)

    def _bcast(self, payloads) -> list:
        owner, index = self._root_of(payloads, 1)
        value = payloads[index][0] if self.rank == owner else None
        if self.ranks > 1:
            value = self.comm.bcast(value, root=owner)
        data = pickle.dumps(value)
        return [payloads[index][0] if (self.rank == owner and position == index) else pickle.loads(data)
                for position in range(self.local)]

    def _copy(self, target, source) -> None:
        """``target <- source`` (contiguous arrays of one size; devices or host), queued on the target's stream."""

        if _is_device(target):
            cp, _ = require_cupy()
            device = int(target.device.id)
            with cp.cuda.Device(device):
                _flat(target).data.copy_from_device_async(_flat(source).data, int(source.nbytes), self.streams[device])
        else:
            np.copyto(_flat(target), _flat(source))

    def _synchronize(self) -> None:
        for stream in self.streams.values():
            stream.synchronize()

    def _Bcast(self, payloads) -> list:
        owner, index = self._root_of(payloads, 1)
        source = payloads[index][0]
        if self.ranks > 1:
            self.comm.Bcast(source, root=owner)
        for position, (buffer, _root) in enumerate(payloads):
            if position != index:
                self._copy(buffer, source)
        self._synchronize()
        return [None] * self.local

    def _Allreduce(self, payloads) -> list:
        in_place = [send is _Shim.IN_PLACE for send, _receive in payloads]
        values = [receive if flag else send for (send, receive), flag in zip(payloads, in_place)]
        target = payloads[0][1]
        if not in_place[0]:
            self._copy(target, values[0])
        if _is_device(target):
            cp, _ = require_cupy()
            device = int(target.device.id)
            stream = self.streams[device]
            flat = _flat(target)
            chunk = max(1, min(int(flat.size), (64 << 20) // 8))
            with cp.cuda.Device(device), stream:
                spare = cp.empty(chunk, dtype=target.dtype)
                for value in values[1:]:
                    source = _flat(value)
                    for start in range(0, int(flat.size), chunk):
                        stop = min(int(flat.size), start + chunk)
                        spare[: stop - start].data.copy_from_device_async(
                            source[start:stop].data, (stop - start) * target.itemsize, stream)
                        flat[start:stop] += spare[: stop - start]
                stream.synchronize()
                del spare
        else:
            for value in values[1:]:
                np.add(target, value, out=target)
        if self.ranks > 1:
            if self._mpi is not None:
                self.comm.Allreduce(self._mpi.IN_PLACE, target)
            elif _is_device(target):
                # A communicator without IN_PLACE (tests): a second array on the target's device.
                cp, _ = require_cupy()
                with cp.cuda.Device(int(target.device.id)):
                    total = target.copy()
                self.comm.Allreduce(target, total)
                self._copy(target, total)
                self._synchronize()
            else:
                total = np.empty_like(target)
                self.comm.Allreduce(target, total)
                np.copyto(target, total)
        for position in range(1, self.local):
            self._copy(payloads[position][1], target)
        self._synchronize()
        return [None] * self.local

    @staticmethod
    def _spec(spec):
        buffer = spec[0]
        counts, displacements = spec[1]
        return _flat(buffer), np.asarray(counts, dtype=np.int64), np.asarray(displacements, dtype=np.int64)

    def _Alltoallv(self, payloads) -> list:
        sends = [self._spec(payload[0]) for payload in payloads]
        receives = [self._spec(payload[1]) for payload in payloads]
        requests = []
        for target, (buffer, counts, displacements) in enumerate(receives):
            for source in range(self.size):
                count = int(counts[source])
                owner, index = divmod(source, self.local)
                if count and owner != self.rank:
                    piece = buffer[int(displacements[source]): int(displacements[source]) + count]
                    requests.append(self._messages.Irecv(piece, source=owner, tag=index * self.local + target))
        remote = 0
        for index, (buffer, counts, displacements) in enumerate(sends):
            for destination in range(self.size):
                count = int(counts[destination])
                if not count:
                    continue
                owner, target = divmod(destination, self.local)
                piece = buffer[int(displacements[destination]): int(displacements[destination]) + count]
                if owner == self.rank:
                    into, into_counts, into_displacements = receives[target]
                    source = self.base + index
                    if int(into_counts[source]) != count:
                        raise RuntimeError("the devices of a rank disagree on the counts of an exchange")
                    start = int(into_displacements[source])
                    self._copy(into[start: start + count], piece)
                else:
                    remote += count * int(piece.itemsize)
                    requests.append(self._messages.Isend(piece, dest=owner, tag=index * self.local + target))
        if self._mpi is not None:
            self._mpi.Request.Waitall(requests)
        else:
            for request in requests:
                request.Wait()
        self.remote_bytes += remote
        self._synchronize()
        return [None] * self.local

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class _DeviceComm:
    """The communicator of one device thread: virtual rank ``base + index`` of the hub's ``R * D``."""

    def __init__(self, hub: _NodeHub, index: int) -> None:
        self.hub = hub
        self.index = int(index)
        self.rank = hub.base + self.index
        self.size = hub.size

    def bcast(self, value, root: int = 0):
        return self.hub.call(self.index, "bcast", (value, int(root)))

    def Bcast(self, buffer, root: int = 0) -> None:
        self.hub.call(self.index, "Bcast", (buffer, int(root)))

    def Allreduce(self, send, receive, op=None) -> None:
        self.hub.call(self.index, "Allreduce", (send, receive))

    def Alltoallv(self, send_spec, receive_spec) -> None:
        self.hub.call(self.index, "Alltoallv", (send_spec, receive_spec))

    def send(self, value, dest: int, tag: int = 0) -> None:
        self.hub.send(self.index, "object", pickle.dumps(value), int(dest), int(tag))

    def recv(self, source: int = 0, tag: int = 0):
        return self.hub.receive(self.index, "object", None, int(source), int(tag))

    def Send(self, buffer, dest: int, tag: int = 0) -> None:
        self.hub.send(self.index, "buffer", buffer, int(dest), int(tag))

    def Recv(self, buffer, source: int = 0, tag: int = 0) -> None:
        self.hub.receive(self.index, "buffer", buffer, int(source), int(tag))


class _VirtualShardGroup(MPIShardGroup):
    """:class:`mpi_shared_basis.MPIShardGroup` of one device of a rank, on the hub's virtual communicator.

    The methods run on the device's thread of the hub; the outer :class:`NestedShardGroup` reports a failure
    (the inner ones do not call the failure handler, which belongs to the calling thread).  The condition and
    eigh helpers of the small solve are kept on the devices of the root's rank: a helper of another rank would
    need a message between ranks outside a collective.
    """

    def __init__(self, hub: _NodeHub, index: int, operator, *, transport: str, ritz_route: str,
                 round_buffers: dict | None) -> None:
        super().__init__(_DeviceComm(hub, index), operator, transport=transport, ritz_route=ritz_route,
                         round_buffers=round_buffers)
        self._mpi = _Shim
        self.hub = hub

    random_basis = MPIShardGroup.random_basis.__wrapped__
    ritz = MPIShardGroup.ritz.__wrapped__
    orthonormal_ritz = MPIShardGroup.orthonormal_ritz.__wrapped__

    def _on_root_rank(self, rank: int | None) -> int | None:
        return rank if rank is not None and rank // self.hub.local == self.owner // self.hub.local else None

    def _condition_helper(self) -> int | None:
        return self._on_root_rank(super()._condition_helper())

    def _eigh_helper(self) -> int | None:
        # The flat rule takes the rank after the condition helper (this class's, which may have been dropped).
        return self._on_root_rank(super()._eigh_helper())


class NestedBasis(SharedBasis):
    """An ``N x m`` basis whose virtual rank ``v`` holds the columns of ``layout.parts[v]``.

    ``blocks`` has one entry per virtual rank of the group, the blocks of this rank's devices at their
    virtual ranks (on their devices) and ``None`` elsewhere, so that code written for the blocks of a
    :class:`distributed_state.DistributedBasis` reaches them through :meth:`NestedShardGroup.map`.
    """

    def __init__(self, group: "NestedShardGroup", blocks, layout) -> None:
        layout = _as_layout(layout)
        if len(layout.parts) != group.size:
            raise ValueError("a nested basis needs the column ranges of every virtual rank")
        blocks = tuple(blocks)
        if len(blocks) != len(group.local_devices):
            raise ValueError("a nested basis needs one block per device of the rank")
        full = [None] * group.size
        rows = set()
        for index, block in enumerate(blocks):
            virtual = group.rank + index
            if (block.ndim != 2 or int(block.shape[1]) != layout.widths[virtual] or not block.flags.f_contiguous
                    or int(block.device.id) != group.local_devices[index]):
                raise ValueError("every block must be Fortran-ordered on its device with the columns of its virtual rank")
            rows.add(int(block.shape[0]))
            full[virtual] = block
        if len(rows) != 1:
            raise ValueError("the blocks of a nested basis must share one row count")
        self.group = group
        self.blocks = tuple(full)
        self.layout = layout
        self.shape = (rows.pop(), layout.columns)

    @property
    def local(self):
        raise AttributeError("a nested basis holds one block per device of the rank (local_blocks)")

    @property
    def local_blocks(self) -> list:
        return [self.blocks[self.group.rank + index] for index in range(len(self.group.local_devices))]

    @property
    def nbytes(self) -> int:
        return sum(int(block.nbytes) for block in self.local_blocks)

    def __getitem__(self, key) -> "NestedBasis":
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != slice(None)
            or not isinstance(key[1], slice)
            or key[1].start not in (None, 0)
            or key[1].step not in (None, 1)
        ):
            raise IndexError("a nested basis supports only basis[:, :count]")
        layout = self.layout.leading(len(range(self.shape[1])[key[1]]))
        return NestedBasis(self.group, [block[:, : layout.widths[self.group.rank + index]]
                                        for index, block in enumerate(self.local_blocks)], layout)

    def take_local(self) -> list:
        """Hand the blocks of this rank's devices to the caller and forget them."""

        blocks = self.local_blocks
        self.blocks = ()
        return blocks

    def density(self, builder, occupations: np.ndarray, volume_element: float) -> np.ndarray:
        """This rank's part of the density of the leading ``len(occupations)`` columns, summed over its devices.

        The parts of the ranks are summed by the density command of the MPI context; a rank that holds none
        of those columns returns zeros.
        """

        weights = np.asarray(occupations, dtype=np.float64)
        if weights.ndim != 1 or weights.size > self.shape[1]:
            raise ValueError("occupations do not select leading columns of this basis")
        selected = self[:, : weights.size]

        def partial(virtual, device):
            block = selected.blocks[virtual]
            if not block.shape[1]:
                return None
            held = [weights[column: column + count] for column, _offset, count in selected.layout.spans(virtual)]
            return builder(block, held[0] if len(held) == 1 else np.concatenate(held), volume_element)

        total = None
        # The devices in their order: the same sum on every run.
        for part in self.group.map(partial):
            if part is not None:
                total = part if total is None else total + part
        if total is None:
            return np.zeros(int(self.shape[0]), dtype=np.float64)
        return total


class NestedShardGroup:
    """The ranks of ``comm`` sharing the basis of one sector, each over the devices ``devices`` of its node.

    ``operator`` is this rank's operator of the sector, on ``devices[0]``; the other devices get replicas
    (:class:`distributed_filter.DistributedFilter`).  Has the methods of :class:`distributed_state.SectorDeviceGroup`
    that :func:`distributed_state.run_distributed_chebff` and :func:`distributed_state.run_distributed_subspace`
    call, with the virtual rank in place of the device index.  ``hubs`` is a dict that the groups of several
    sectors of one communicator share for their hub (solved one after another); ``round_buffers`` a dict of
    per-device dicts of round buffers that they share likewise.
    """

    owner = 0

    def __init__(self, comm, operator, devices, *, transport: str | None = None, ritz_route: str = "rows",
                 round_buffers: dict | None = None, hubs: dict | None = None) -> None:
        devices = tuple(int(device) for device in devices)
        owner_device = int(operator.effective_potential.device.id)
        if len(devices) < 2 or devices[0] != owner_device:
            raise ValueError("a nested group spreads a share over two or more devices, the operator's first")
        worker = getattr(operator, "_distributed_filter", None)
        if worker is None or tuple(worker.devices) != devices:
            worker = DistributedFilter(operator, devices)
            operator._distributed_filter = worker
        self._worker = worker
        self._owner_ref = ref(operator)
        self.transport = shared_transport() if transport is None else transport
        hubs = {} if hubs is None else hubs
        hub = hubs.get(id(comm))
        if hub is None or hub.devices != devices or hub.transport != self.transport:
            hub = _NodeHub(comm, devices, self.transport)
            hubs[id(comm)] = hub
        self.hub = hub
        self.comm = comm
        self.ritz_route = ritz_route
        self.size = hub.size
        self.rank = hub.base
        self.devices = tuple(range(self.size))
        self.local_devices = devices
        self.device = devices[0]
        buffers = {} if round_buffers is None else round_buffers
        self._groups = [
            _VirtualShardGroup(hub, index, self._operator(device), transport=self.transport,
                               ritz_route=ritz_route, round_buffers=buffers.setdefault(device, {}))
            for index, device in enumerate(devices)
        ]
        self.filter_seconds = 0.0

    # ----- reports -----------------------------------------------------------------------------------------------

    @property
    def passes(self) -> int:
        return int(self._groups[0].passes)

    @property
    def exchanged_bytes(self) -> int:
        return int(sum(group.exchanged_bytes for group in self._groups))

    @property
    def separate_row_blocks(self) -> int:
        return int(sum(group.separate_row_blocks for group in self._groups))

    @property
    def seconds(self) -> dict:
        """The most seconds of a device of the rank in each stage, the filter, and the hub's MPI seconds."""

        seconds = {name: max(group.seconds[name] for group in self._groups) for name in self._groups[0].seconds}
        seconds["filter"] = self.filter_seconds
        seconds.update({f"mpi_{name}": value for name, value in self.hub.seconds.items()})
        return seconds

    @property
    def remote_bytes(self) -> int:
        return int(self.hub.remote_bytes)

    # ----- the interface of SectorDeviceGroup -----------------------------------------------------------------

    def _operator(self, device=None):
        if device is None or int(device) == self.device:
            operator = self._owner_ref()
            if operator is None:
                raise ReferenceError("the operator of a nested group was released")
            return operator
        return self._worker.replicas[int(device)]

    def map(self, function):
        """Run ``function(virtual rank, device)`` for every device of this rank, on its thread and stream."""

        cp, _ = require_cupy()

        def job(index, device):
            stream = self._worker.streams[device]

            def run():
                with cp.cuda.Device(device), stream:
                    value = function(self.rank + index, device)
                    stream.synchronize()
                    return value

            return run

        outcomes = self.hub.run([job(index, device) for index, device in enumerate(self.local_devices)])
        for ok, value in outcomes:
            if not ok:
                raise value
        return [value for _ok, value in outcomes]

    def column_layout(self, blocks):
        return self._groups[0].column_layout(blocks)

    def _device_jobs(self, function):
        """One job per device of the rank: ``function(index, group)`` under its device and stream."""

        cp, _ = require_cupy()

        def job(index, group):
            device = self.local_devices[index]
            stream = self._worker.streams[device]

            def run():
                with cp.cuda.Device(device), stream:
                    return function(index, group)

            return run

        return [job(index, group) for index, group in enumerate(self._groups)]

    @staticmethod
    def _raise_first(outcomes) -> None:
        failures = [value for ok, value in outcomes if not ok]
        if not failures:
            return
        for failure in failures:
            if not isinstance(failure, (_HubFailure, GeneralizedRitzStabilityError)):
                raise failure
        raise failures[0]

    @_ends_the_job_on_failure
    def random_basis(self, generator, rows: int, layout) -> NestedBasis:
        """The columns of PARSEC's DLARNV stream of every device of this rank (those of the one-device basis)."""

        layout = _as_layout(layout)
        rows = int(rows)
        start = generator.seed

        def draw(index, group):
            # Each device skips ahead in its own copy of the stream.
            basis = group.random_basis(copy.copy(generator), rows, layout)
            return basis.take()[group.rank]

        outcomes = self.hub.run(self._device_jobs(draw))
        self._raise_first(outcomes)
        generator.seed = _skip(start, rows * layout.columns)
        return NestedBasis(self, [value for _ok, value in outcomes], layout)

    @_ends_the_job_on_failure
    def fill_leading(self, basis: NestedBasis, saved) -> None:
        """Write saved columns into the leading columns on every device of this rank (:mod:`sector_growth`)."""

        layout = basis.layout
        blocks = basis.local_blocks

        def fill(index, group):
            group._fill_block(blocks[index], layout, saved)

        self._raise_first(self.hub.run(self._device_jobs(fill)))

    @_ends_the_job_on_failure
    def filter(self, basis: NestedBasis, blocks, lower, upper, reference, reset) -> NestedBasis:
        """Chebyshev-filter the columns of every device in place with the root's plan and interval."""

        cp, _ = require_cupy()
        started = perf_counter()
        plan = (tuple(blocks), float(lower), float(upper), float(reference))
        if self.hub.ranks > 1:
            plan = self.comm.bcast(plan, root=0)
        blocks, lower, upper, reference = plan
        layout = basis.layout
        partitions = SectorDeviceGroup._whole_block_partitions(layout, blocks)
        if partitions is None:
            raise RuntimeError("the columns of a nested basis are no whole filter blocks of this plan")
        sigmas = starting_sigmas(blocks, lower, upper, reset, reference)
        owner = self._operator()
        with cp.cuda.Device(self.device):
            cp.cuda.get_current_stream().synchronize()
            potential = cp.asnumpy(owner.effective_potential)
        columns = basis.take_local()

        def run(index, group):
            device = self.local_devices[index]
            block = columns[index]
            mine = partitions[self.rank + index]
            if not mine:
                return block
            stream = self._worker.streams[device]
            if device != self.device:
                self._worker.replicas[device].effective_potential.set(potential, stream=stream)
            local, starts, offset = [], [], 0
            for first, last in mine:
                shift = offset - blocks[first].start
                local.extend(FilterBlock(b.start + shift, b.stop + shift, b.degree) for b in blocks[first:last])
                starts.extend(sigmas[first:last])
                offset = local[-1].stop
            result = self._worker.graphs[device].apply(
                block, tuple(local), lower, upper, reference, reset, initial_sigma=starts, out=block)
            stream.synchronize()
            return result

        outcomes = self.hub.run(self._device_jobs(run))
        self._raise_first(outcomes)
        self.filter_seconds += perf_counter() - started
        return NestedBasis(self, [value for _ok, value in outcomes], layout)

    def _ritz_outcome(self, outcomes, layout):
        failures = [value for ok, value in outcomes if not ok]
        if failures:
            stability = [failure for failure in failures if isinstance(failure, GeneralizedRitzStabilityError)]
            if len(stability) == len(outcomes):
                filtered = NestedBasis(self, [failure.filtered.take()[group.rank]
                                              for failure, group in zip(stability, self._groups)], layout)
                message = str(stability[0])
                del stability, failures
                raise _stability_error(message, filtered)
            self._raise_first(outcomes)
        values = [value for _ok, value in outcomes]
        eigenvalues, _basis, whitened = values[0]
        blocks = [value[1].take()[group.rank] for value, group in zip(values, self._groups)]
        return eigenvalues, NestedBasis(self, blocks, layout), whitened

    @_ends_the_job_on_failure
    def ritz(self, operator: Any, filtered: NestedBasis, *, stages: bool):
        """Generalized Rayleigh--Ritz of the filtered basis over every device of every rank; consumes ``filtered``.

        Returns the Ritz values (on every rank), the rotated basis and the whitened projected Hamiltonian (on
        the root; an empty array elsewhere).  On a failed stability audit raises
        :class:`rayleigh_ritz.GeneralizedRitzStabilityError` with the intact filtered basis as ``error.filtered``.
        """

        layout = filtered.layout
        columns = filtered.take_local()

        def run(index, group):
            local = operator if index == 0 else group._operator()
            return group.ritz(local, SharedBasis(group, columns[index], layout), stages=bool(stages) and index == 0)

        outcomes = self.hub.run(self._device_jobs(run))
        del columns
        return self._ritz_outcome(outcomes, layout)

    @_ends_the_job_on_failure
    def orthonormal_ritz(self, operator: Any, filtered: NestedBasis, generator) -> tuple:
        """The fallback of :meth:`mpi_shared_basis.MPIShardGroup.orthonormal_ritz` on the virtual ranks."""

        layout = filtered.layout
        columns = filtered.take_local()

        def run(index, group):
            local = operator if index == 0 else group._operator()
            return group.orthonormal_ritz(local, SharedBasis(group, columns[index], layout), generator)

        outcomes = self.hub.run(self._device_jobs(run))
        del columns
        return self._ritz_outcome(outcomes, layout)


def attach_nested_group(operator, comm, devices, *, ritz_route: str = "rows", round_buffers: dict | None = None,
                        hubs: dict | None = None, transport: str | None = None) -> NestedShardGroup:
    """Give ``operator`` (this rank's operator of the shared sector, on ``devices[0]``) its nested group.

    The group takes the place of :func:`mpi_shared_basis.attach_shared_group` where a rank spreads its share
    over the devices of its node (:func:`experimental.mpi_scf.nested_sharing_applies`).
    """

    group = NestedShardGroup(comm, operator, devices, transport=transport, ritz_route=ritz_route,
                             round_buffers=round_buffers, hubs=hubs)
    operator._mpi_shared_group = group
    return group


__all__ = ["NestedBasis", "NestedShardGroup", "attach_nested_group"]
