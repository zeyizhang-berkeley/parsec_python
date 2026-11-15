"""Bounded scratch storage for dense FP64/complex128 steps and collectives.

The scientific arrays retain their dtype. Only temporary storage and data
placement change; no compression, mixed precision or weaker audits are used.
The same helpers work with NumPy and CuPy and arbitrary communicator sizes.
"""
from __future__ import annotations

import math
import os

import numpy as np


def buffer_bytes() -> int:
    """Maximum bytes per dense communication buffer (64 MiB by default)."""
    value = int(os.environ.get("PARSEC_DENSE_BUFFER_BYTES", str(64 * 1024**2)))
    if value < 16:
        raise ValueError("PARSEC_DENSE_BUFFER_BYTES must be at least 16")
    return value


def _tile_width(dtype) -> int:
    return max(1, math.isqrt(buffer_bytes() // np.dtype(dtype).itemsize))


def permute_columns_in_place(matrix, order, xp):
    """Set ``matrix[:, j] = old_matrix[:, order[j]]`` in bounded row tiles.

    The scratch is one row tile, at most ``buffer_bytes()`` or one complete
    row when that is larger, plus an O(columns) integer permutation. The
    matrix keeps its allocation, dtype and strides. This also works on a
    view inside the padded column/row storage used by MPI.
    """
    indices = np.asarray(order)
    if (matrix.ndim != 2 or indices.ndim != 1
            or indices.size != matrix.shape[1] or indices.dtype.kind not in "iu"
            or not np.array_equal(np.sort(indices), np.arange(indices.size))):
        raise ValueError("order must be a permutation of the matrix columns")
    rows, columns = map(int, matrix.shape)
    if not rows or not columns or np.array_equal(indices, np.arange(columns)):
        return matrix
    height = max(1, buffer_bytes() // (columns * matrix.dtype.itemsize))
    device_order = xp.asarray(indices, dtype=xp.int64)
    for first in range(0, rows, height):
        last = min(rows, first + height)
        # Advanced indexing finishes the gather before overwriting this tile;
        # no source row outside the tile is ever read or changed.
        matrix[first:last, :] = matrix[first:last, device_order]
    return matrix


def mirror_lower(matrix, xp, *, copy=True):
    """Mirror the lower triangle with bounded tiles; never read the upper one.

    With ``copy=False`` the caller gives up the original matrix. No full-sized
    triangle, index array or transpose copy is allocated. Complex entries are
    conjugated, preserving the Hermitian matrix used by the k-point solver.
    """
    full = matrix.copy(order="C") if copy else matrix
    count = int(full.shape[0])
    width = _tile_width(full.dtype)
    for first in range(0, count, width):
        last = min(count, first + width)
        diagonal = full[first:last, first:last]
        lower = xp.tril(diagonal)
        diagonal[...] = lower + xp.tril(lower, -1).conj().T
        del lower
        for start in range(last, count, width):
            stop = min(count, start + width)
            full[first:last, start:stop] = full[start:stop, first:last].conj().T
    return full


def sum_in_place(array, xp, allreduce, *, transport):
    """Consume a C-contiguous array, summing bounded chunks into its storage.

    ``allreduce(send, receive)`` is the communicator's collective. Host staging
    copies directly into the existing device allocation, avoiding an upload
    allocation of the whole result. Every rank must use the same byte budget.
    """
    if not array.flags.c_contiguous:
        raise ValueError("an in-place dense sum requires C-contiguous storage")
    flat = array.reshape(-1)
    step = max(1, buffer_bytes() // array.dtype.itemsize)
    for first in range(0, flat.size, step):
        piece = flat[first:first + step]
        if xp is np or transport == "cuda":
            result = xp.empty_like(piece)
            if xp is not np:
                # MPI may reuse the previous receive allocation; its queued
                # device-to-device copy must finish before another reduction.
                xp.cuda.get_current_stream().synchronize()
            allreduce(piece, result)
            piece[...] = result
        else:
            source = xp.asnumpy(piece)
            result = np.empty_like(source)
            allreduce(source, result)
            piece.set(result)
            # Keep pageable host storage alive until its transfer completes.
            xp.cuda.get_current_stream().synchronize()
        del result
    return array


def lower_sum_in_place(matrix, xp, reduce, *, root, transport):
    """Consume local lower triangles; leave their sum in the root's matrix.

    Rectangular tiles below the diagonal and square diagonal tiles travel in
    bounded buffers. This removes the two O(m²) index arrays and packed lower
    triangle used by advanced indexing. Non-root ranks return None. The root
    gets zeros above the diagonal, as the previous packed reduction did.
    """
    count = int(matrix.shape[0])
    width = _tile_width(matrix.dtype)
    for first in range(0, count, width):
        last = min(count, first + width)
        for start in range(0, last, width):
            stop = min(last, start + width)
            piece = matrix[first:last, start:stop]
            send = xp.ascontiguousarray(piece)
            if xp is np or transport == "cuda":
                receive = xp.empty_like(send) if root else None
                if xp is not np:
                    xp.cuda.get_current_stream().synchronize()
                reduce(send, receive)
                if root:
                    piece[...] = receive
            else:
                host = xp.asnumpy(send)
                receive = np.empty_like(host) if root else None
                reduce(host, receive)
                if root:
                    # A tile can be strided; only this bounded tile is uploaded.
                    piece[...] = xp.asarray(receive)
                del host
            del send, receive
        if root:
            matrix[first:last, first:last] = xp.tril(matrix[first:last, first:last])
            matrix[first:last, last:] = 0
    return matrix if root else None


def broadcast_matrix(matrix, shape, dtype, xp, bcast, *, root, transport):
    """Broadcast into the final Fortran-ordered storage, a bounded chunk at a time."""
    if root and isinstance(matrix, xp.ndarray):
        result = xp.asfortranarray(matrix, dtype=dtype)
    else:
        result = xp.empty(shape, dtype=dtype, order="F")
    flat = result.T.reshape(-1)  # contiguous view, not a second matrix
    host_source = None
    if root and not isinstance(matrix, xp.ndarray):
        host_source = np.asfortranarray(matrix, dtype=dtype).T.reshape(-1)
    step = max(1, buffer_bytes() // np.dtype(dtype).itemsize)
    for first in range(0, flat.size, step):
        piece = flat[first:first + step]
        if xp is np or transport == "cuda":
            if host_source is not None:
                piece[...] = xp.asarray(host_source[first:first + step])
            if xp is not np:
                xp.cuda.get_current_stream().synchronize()
            bcast(piece)
        else:
            host = (np.ascontiguousarray(host_source[first:first + step]) if host_source is not None
                    else xp.asnumpy(piece)) if root else np.empty(piece.size, dtype=dtype)
            bcast(host)
            if not root or host_source is not None:
                piece.set(host)
                xp.cuda.get_current_stream().synchronize()
            del host
    return result
