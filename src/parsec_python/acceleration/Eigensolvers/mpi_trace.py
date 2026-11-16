"""Opt-in, all-rank metadata for existing MPI exchanges; never inspect payloads.

PARSEC_MPI_TRACE=1 enables flushed JSON independently of PARSEC_DEBUG. No MPI
or device calls are made here. Group identifiers are process-local instances;
combine them with world/local ranks when reconstructing communicator membership.
Only 256 layout signatures are retained per group, not array references.
"""
from collections import OrderedDict
import hashlib
import itertools
import json
from numbers import Integral
import os
import sys
import threading
import time

_groups = itertools.count()
_events = itertools.count(1)
_lock = threading.Lock()
_error_categories = set()


def _rank():
    for key in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def _trace_error(category, error):
    """Best-effort, once-per-category failure notice without exception payloads."""
    try:
        with _lock:
            if category in _error_categories:
                return
            _error_categories.add(category)
            try:
                rank = _rank()
            except Exception:
                rank = None
            record = dict(stage="trace_error", rank=rank, time=time.time(),
                          monotonic_ns=time.monotonic_ns(), error_type=type(error).__name__)
            print("PARSEC_MPI_TRACE " + json.dumps(record, separators=(",", ":")), flush=True)
    except Exception:
        # A failed output sink cannot be repaired by additional logging or MPI.
        pass


def _group_device_id(group):
    value = getattr(group, "device", None)
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    _trace_error("group_device_id", TypeError())
    return None


def _buffer(array):
    dtype = array.dtype
    flags = array.flags
    device = getattr(array, "device", None)
    return dict(dtype=str(dtype), shape=[int(n) for n in array.shape],
                size=int(array.size), nbytes=int(array.nbytes),
                c_contiguous=bool(flags.c_contiguous), f_contiguous=bool(flags.f_contiguous),
                device_id=int(device.id) if device is not None and hasattr(device, "id") else None)


def _offsets(counts):
    result, total = [], 0
    for count in counts:
        result.append(total)
        total += count
    return result


class ExchangeTrace:
    def __init__(self, identity):
        self.identity = identity

    def emit(self, stage, **values):
        # A broken diagnostic sink must not strand peers in an existing collective.
        try:
            record = dict(self.identity, stage=stage, time=time.time(),
                          monotonic_ns=time.monotonic_ns(), event_sequence=next(_events), **values)
            line = json.dumps(record, allow_nan=False, separators=(",", ":"))
            with _lock:
                print("PARSEC_MPI_TRACE " + line, flush=True)
        except Exception as error:
            _trace_error("emit", error)


def start_exchange(group, send, send_counts, receive, receive_counts, *, wire_multiplier=1, label=None):
    if os.environ.get("PARSEC_MPI_TRACE", "off").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    try:
        if label is None:
            # Identify direct QR/Ritz/redistribution call sites without changing
            # callers (including custom four-argument exchange implementations).
            caller = sys._getframe(2)
            label = caller.f_code.co_name + ":" + str(caller.f_lineno)
            del caller
        state = getattr(group, "_mpi_trace_state", None)
        if state is None:
            state = dict(group_id=type(group).__name__ + ":" + str(next(_groups)),
                         sequence=0, layouts=OrderedDict())
            group._mpi_trace_state = state
        state["sequence"] += 1
        logical_sends = [int(n) for n in send_counts]
        logical_receives = [int(n) for n in receive_counts]
        sends = [n * wire_multiplier for n in logical_sends]
        receives = [n * wire_multiplier for n in logical_receives]
        s_offsets, r_offsets = _offsets(sends), _offsets(receives)
        send_info, receive_info = _buffer(send), _buffer(receive)
        layout = dict(send=send_info, receive=receive_info,
            logical_send_counts=logical_sends, logical_receive_counts=logical_receives,
            send_counts=sends, receive_counts=receives,
            send_displacements=s_offsets, receive_displacements=r_offsets,
            wire_dtype="float64", wire_itemsize=8, wire_multiplier=wire_multiplier,
            send_bytes=8 * sum(sends), receive_bytes=8 * sum(receives),
            counts_valid=(len(sends) == group.size and len(receives) == group.size
                          and all(n >= 0 for n in sends + receives)),
            send_bounds_valid=all(n >= 0 and 8 * (d + n) <= send_info["nbytes"]
                                  for n, d in zip(sends, s_offsets)),
            receive_bounds_valid=all(n >= 0 and 8 * (d + n) <= receive_info["nbytes"]
                                     for n, d in zip(receives, r_offsets)))
        digest = hashlib.sha256(json.dumps(layout, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        identity = dict(schema="parsec-mpi-trace-v1", rank=_rank(), local_rank=int(group.rank),
            comm_rank=int(group.rank),
            comm_size=int(group.size), group_id=state["group_id"],
            group_id_scope="process_local_instance", group_sequence=state["sequence"],
            transport=group.transport, label=label, layout_signature=digest,
            group_device_id=_group_device_id(group))
        trace = ExchangeTrace(identity)
        if digest not in state["layouts"]:
            trace.emit("layout", **layout)
            state["layouts"][digest] = None
            if len(state["layouts"]) > 256:
                state["layouts"].popitem(last=False)
        return trace
    except Exception as error:
        # Metadata collection is observational; it must not change solver behavior.
        _trace_error("metadata", error)
        return None
