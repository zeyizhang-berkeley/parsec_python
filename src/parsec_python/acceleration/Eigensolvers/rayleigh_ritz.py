"""Device-resident Rayleigh--Ritz projection and rotation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from ..backends.cupy import device_stage, require_cupy
from .small_dense import symmetric_eigh


_DEFAULT_GENERALIZED_RITZ_WORK_THRESHOLD = 100_000_000


class GeneralizedRitzStabilityError(np.linalg.LinAlgError):
    """Raised when the non-orthogonal Ritz basis fails a safety audit."""


@dataclass(frozen=True)
class DeviceRayleighRitzResult:
    eigenvalues: Any
    wavefunctions: Any
    applied_wavefunctions: Any | None
    projected_hamiltonian: Any
    residual_norms: Any | None
    workspace: Any | None = None
    algorithm: str = "orthonormal_rayleigh_ritz"


def generalized_ritz_requested(row_count: int, column_count: int) -> bool:
    """Return whether the audited non-orthogonal Ritz route is selected.

    The route is profitable only for a large complete basis.  Explicitly
    selecting an orthogonalization algorithm remains a source-comparison
    request and therefore disables automatic generalized Ritz unless this
    policy is forced ``on`` separately.
    """

    policy = os.environ.get("PARSEC_CUPY_GENERALIZED_RITZ", "auto")
    policy = policy.strip().lower()
    if policy not in {"auto", "on", "off", "1", "0", "true", "false"}:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ must be auto, on, or off"
        )
    if policy in {"on", "1", "true"}:
        return True
    if policy in {"off", "0", "false"}:
        return False
    explicit_orthogonalization = os.environ.get(
        "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION"
    )
    if explicit_orthogonalization is not None and (
        explicit_orthogonalization.strip().lower() != "auto"
    ):
        return False
    raw_threshold = os.environ.get(
        "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD",
        str(_DEFAULT_GENERALIZED_RITZ_WORK_THRESHOLD),
    ).strip()
    try:
        threshold = int(raw_threshold)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD must be an integer"
        ) from error
    if threshold < 0:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD cannot be negative"
        )
    work = int(row_count) * int(column_count) * int(column_count)
    return work >= threshold


def _generalized_condition_limit() -> float:
    raw = os.environ.get(
        "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX", "1.0e8"
    ).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX must be numeric"
        ) from error
    if not np.isfinite(value) or value <= 1.0:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX must exceed one"
        )
    return value


def _symmetric_overlap(matrix: Any):
    """Form the lower triangle of ``X.T X``, optionally with FP64 DSYRK."""

    cp, _ = require_cupy()
    policy = os.environ.get(
        "PARSEC_CUPY_RITZ_SYRK", "auto"
    ).strip().lower()
    if policy not in {"auto", "on", "off", "1", "0", "true", "false"}:
        raise ValueError("PARSEC_CUPY_RITZ_SYRK must be auto, on, or off")
    if policy in {"off", "0", "false"}:
        return matrix.T @ matrix
    # cuBLAS is column-major.  ``matrix`` is made Fortran-contiguous by the
    # caller, so op(A)=A.T with n=states and k=grid points computes exactly
    # the lower triangle consumed by the generalized Ritz solve.
    from cupy.cuda import cublas

    rows, columns = map(int, matrix.shape)
    try:
        output = cp.empty(
            (columns, columns), dtype=cp.float64, order="F"
        )
        alpha = np.asarray(1.0, dtype=np.float64)
        beta = np.asarray(0.0, dtype=np.float64)
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        cublas.dsyrk(
            handle,
            cublas.CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_T,
            columns,
            rows,
            alpha.ctypes.data,
            matrix.data.ptr,
            rows,
            beta.ctypes.data,
            output.data.ptr,
            columns,
        )
        return output
    except Exception:
        if policy in {"on", "1", "true"}:
            raise
        return matrix.T @ matrix


def generalized_rayleigh_ritz(
    operator: Any,
    basis: Any,
    *,
    workspace: Any | None = None,
    compute_residuals: bool = False,
) -> DeviceRayleighRitzResult:
    """Solve Rayleigh--Ritz directly in a non-orthogonal filtered basis.

    For filtered columns ``X``, this routine solves

    ``(X.T H X) C = (X.T X) C diag(epsilon)``.

    A host Cholesky factor of the small overlap matrix whitens the generalized
    problem.  This is algebraically equivalent to orthonormalizing ``X`` and
    applying ordinary Rayleigh--Ritz, but avoids a tall Householder QR.  The
    overlap condition number, Cholesky factorization, and final coefficient
    orthogonality are audited.  Callers must catch
    :class:`GeneralizedRitzStabilityError` and use robust QR when an unusually
    ill-conditioned filtered basis fails any audit.

    ``workspace`` is a persistent device ``N x m`` array receiving ``H X``.
    Reusing it avoids a costly large allocation/memory-pool eviction in every
    nonlinear iteration.
    """

    cp, _ = require_cupy()
    matrix = cp.asarray(basis, dtype=cp.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("basis must be a nonempty two-dimensional matrix")
    if tuple(operator.shape) != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("operator shape must match basis rows")
    # SUBSPACE assembles independently filtered blocks into a row-major
    # array.  The real-space CUDA kernel traverses one orbital down the grid,
    # and the tall Gram/rotation GEMMs likewise favor column-contiguous
    # storage.  A single device copy (about 0.03 s for Si28H36) avoids strided
    # reads in all three much larger operations; it does not change a value.
    if not matrix.flags.f_contiguous:
        matrix = cp.asfortranarray(matrix)
    if (
        workspace is None
        or not isinstance(workspace, cp.ndarray)
        or workspace.dtype != cp.dtype(cp.float64)
        or workspace.shape != matrix.shape
    ):
        workspace = cp.empty(matrix.shape, dtype=cp.float64, order="F")

    with device_stage(operator, "subspace_ritz_hamiltonian_seconds"):
        apply_into = getattr(operator, "apply_into", None)
        if callable(apply_into):
            applied_basis = apply_into(matrix, workspace)
        else:
            workspace[...] = operator @ matrix
            applied_basis = workspace

    with device_stage(operator, "subspace_ritz_projection_seconds"):
        raw_overlap = _symmetric_overlap(matrix)
        raw_projection = matrix.T @ applied_basis
        packed = np.asarray(
            cp.asnumpy(cp.stack((raw_overlap, raw_projection))),
            dtype=np.float64,
        )
    overlap = np.tril(packed[0]) + np.tril(packed[0], -1).T
    projected = np.tril(packed[1]) + np.tril(packed[1], -1).T
    try:
        condition = float(np.linalg.cond(overlap))
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap condition estimate failed"
        ) from error
    if not np.isfinite(condition) or condition > _generalized_condition_limit():
        raise GeneralizedRitzStabilityError(
            f"filtered overlap condition number {condition:.3e} is unsafe"
        )
    try:
        cholesky = np.linalg.cholesky(overlap)
        left_projected = solve_triangular(
            cholesky,
            projected,
            lower=True,
            check_finite=False,
        )
        whitened = solve_triangular(
            cholesky,
            left_projected.T,
            lower=True,
            check_finite=False,
        ).T
        whitened = np.tril(whitened) + np.tril(whitened, -1).T
        host_eigenvalues, whitened_vectors = np.linalg.eigh(whitened)
        coefficients = solve_triangular(
            cholesky.T,
            whitened_vectors,
            lower=False,
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap Cholesky/Ritz solve failed"
        ) from error

    coefficient_overlap = coefficients.T @ overlap @ coefficients
    audit_error = float(
        np.max(
            np.abs(
                coefficient_overlap
                - np.eye(coefficient_overlap.shape[0], dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    if not np.isfinite(audit_error) or audit_error > 5.0e-10:
        raise GeneralizedRitzStabilityError(
            f"generalized Ritz orthogonality audit failed ({audit_error:.3e})"
        )

    device_coefficients = cp.asarray(coefficients, dtype=cp.float64)
    eigenvalues = cp.asarray(host_eigenvalues, dtype=cp.float64)
    with device_stage(operator, "subspace_ritz_rotation_seconds"):
        wavefunctions = matrix @ device_coefficients
    if compute_residuals:
        applied_wavefunctions = applied_basis @ device_coefficients
        residuals = (
            applied_wavefunctions
            - wavefunctions * eigenvalues[None, :]
        )
        residual_norms = cp.linalg.norm(residuals, axis=0)
    else:
        applied_wavefunctions = None
        residual_norms = None
    return DeviceRayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=wavefunctions,
        applied_wavefunctions=applied_wavefunctions,
        projected_hamiltonian=cp.asarray(whitened, dtype=cp.float64),
        residual_norms=residual_norms,
        workspace=workspace,
        algorithm="generalized_cholesky_rayleigh_ritz",
    )


def rayleigh_ritz(
    operator: Any,
    basis: Any,
    *,
    compute_residuals: bool = True,
) -> DeviceRayleighRitzResult:
    """Compute a device Rayleigh--Ritz projection and rotation.

    ``CHEBFF`` only needs the Ritz values and rotated vectors to update its
    next filter interval.  PARSEC's CHEBFF path likewise does not form Ritz
    residuals, so ``compute_residuals=False`` skips the extra grid-by-state
    ``(H Q) C`` multiplication and residual reduction.  SUBSPACE keeps the
    default because its residual norms are exposed as iteration diagnostics.
    """

    cp, _ = require_cupy()
    basis = cp.asarray(basis, dtype=cp.float64)
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("basis must be a nonempty two-dimensional matrix")
    if tuple(operator.shape) != (basis.shape[0], basis.shape[0]):
        raise ValueError("operator shape must match basis rows")

    applied_basis = operator @ basis
    if applied_basis.shape != basis.shape:
        raise ValueError("operator must preserve the basis shape")
    raw_projection = basis.T @ applied_basis
    lower = cp.tril(raw_projection)
    projected = lower + cp.tril(lower, -1).T
    eigenvalues, rotations = symmetric_eigh(projected)
    wavefunctions = basis @ rotations
    if compute_residuals:
        applied_wavefunctions = applied_basis @ rotations
        residuals = applied_wavefunctions - wavefunctions * eigenvalues[None, :]
        residual_norms = cp.linalg.norm(residuals, axis=0)
    else:
        applied_wavefunctions = None
        residual_norms = None
    return DeviceRayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=wavefunctions,
        applied_wavefunctions=applied_wavefunctions,
        projected_hamiltonian=projected,
        residual_norms=residual_norms,
        algorithm="orthonormal_rayleigh_ritz",
    )


__all__ = [
    "DeviceRayleighRitzResult",
    "GeneralizedRitzStabilityError",
    "generalized_rayleigh_ritz",
    "generalized_ritz_requested",
    "rayleigh_ritz",
]
