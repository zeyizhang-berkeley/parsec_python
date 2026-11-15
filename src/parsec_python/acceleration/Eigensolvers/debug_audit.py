"""Read-only, bounded-column Ritz residual audits for shared real/complex bases."""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from parsec_python import diagnostics as debug


def column_audit(operator, block, indices, eigenvalues, xp, budget):
    """Measure Hq-eq and diagonal Q*Q without another tall basis.

    ``operator.apply_into`` must have the usual non-mutating-input contract.
    Three one-column arrays are the irreducible scratch floor if a column
    is larger than the requested budget. No vector is changed or returned.
    """
    rows, columns = map(int, block.shape)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (columns,):
        raise ValueError("debug column indices differ from the local basis")
    width = max(1, int(budget) // (3 * max(rows, 1) * block.dtype.itemsize))
    maximum, norm_error, square_sum, worst = 0., 0., 0., None
    finite = True
    for first in range(0, columns, width):
        last = min(first + width, columns)
        vectors = block[:, first:last]
        work = xp.empty(vectors.shape, dtype=block.dtype, order="F")
        operator.apply_into(vectors, work)
        work -= vectors * xp.asarray(eigenvalues[indices[first:last]])[None, :]
        residuals = xp.sqrt(xp.sum((work.conj() * work).real, axis=0))
        norms = xp.sum((vectors.conj() * vectors).real, axis=0)
        host = np.asarray(residuals) if xp is np else xp.asnumpy(residuals)
        norm_host = np.asarray(norms) if xp is np else xp.asnumpy(norms)
        finite = finite and bool(np.isfinite(host).all() and np.isfinite(norm_host).all())
        if host.size:
            index = int(np.argmax(host))
            if worst is None or host[index] > maximum:
                maximum, worst = float(host[index]), int(indices[first + index])
            norm_error = max(norm_error, float(np.max(np.abs(norm_host - 1.))))
            square_sum += float(np.dot(host, host))
        del work, residuals, norms
    return dict(columns=columns, maximum=maximum, squared_sum=square_sum,
                worst_state_zero_based=worst, norm_squared_error=norm_error, finite=finite,
                scratch_floor_bytes=3 * rows * block.dtype.itemsize,
                column_tile=min(columns, width))


def shared_ritz_audit(group, basis, eigenvalues, *, phase: str):
    """All participating ranks enter this only under uniform PARSEC_DEBUG=audit."""
    if not debug.enabled(2):
        return None
    from ..backends.cupy import require_cupy

    xp, _ = require_cupy()
    started = perf_counter()
    values = np.asarray(eigenvalues)

    def inspect(index, device):
        block = basis.blocks[index]
        if block is None:
            return None
        pieces = [np.arange(start, start + count, dtype=np.int64)
                  for start, _offset, count in basis.layout.spans(index)]
        columns = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
        return column_audit(group._operator(device), block, columns, values, xp, debug.buffer_bytes())

    parts = [part for part in group.map(inspect) if part is not None]
    comm = getattr(group, "comm", None)
    if comm is not None:
        parts = [part for rank_parts in comm.allgather(parts) for part in rank_parts]
    count = sum(p["columns"] for p in parts)
    worst = max(parts, key=lambda p: p["maximum"], default=None)
    report = dict(phase=phase, checked_states=count, expected_states=int(basis.shape[1]),
                  all_finite=all(p["finite"] for p in parts),
                  maximum_residual_ry=None if worst is None else worst["maximum"],
                  rms_residual_ry=math.sqrt(sum(p["squared_sum"] for p in parts) / max(1, count)),
                  worst_state_zero_based=None if worst is None else worst["worst_state_zero_based"],
                  maximum_norm_squared_error=max((p["norm_squared_error"] for p in parts), default=0.),
                  scratch_budget_bytes=debug.buffer_bytes(),
                  scratch_floor_bytes=max((p["scratch_floor_bytes"] for p in parts), default=0),
                  seconds=perf_counter() - started,
                  note="Diagnostic only; no convergence decision. Column norms do not audit off-diagonal orthogonality.")
    leader = comm is None or group.rank == group.owner
    debug.emit("eigenpair_audit", minimum=2, leader=leader, **report)
    return report
