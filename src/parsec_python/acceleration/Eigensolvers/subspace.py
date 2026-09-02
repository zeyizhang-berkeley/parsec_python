"""One device-resident later-SCF PARSEC SUBSPACE update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.subspace import SubspaceSettings

from ..backends.cupy import device_stage
from .chebyshev import FilterBlock, subspace_filter, subspace_filter_blocks
from .orthogonalize import orthonormalize_complete_subspace
from .rayleigh_ritz import (
    DeviceRayleighRitzResult,
    GeneralizedRitzStabilityError,
    generalized_rayleigh_ritz,
    generalized_ritz_requested,
    rayleigh_ritz,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class DeviceSubspaceState:
    """Buffered Ritz state kept on the GPU across nonlinear iterations."""

    operator_dimension: int
    working_states: int
    eigenvalues: Any
    vectors: Any
    filter_lower_bound: float | None = None
    first_filter: bool = True
    filters_completed: int = 0
    ritz_workspace: Any | None = None
    generalized_ritz_failed: bool = False


@dataclass(frozen=True)
class DeviceSubspaceResult:
    eigenvalues: Any
    vectors: Any
    residual_norms: Any | None
    state: DeviceSubspaceState
    lanczos_bound: LanczosBoundResult
    polynomial_degree_used: int
    filter_blocks: tuple[FilterBlock, ...]
    rayleigh_ritz: DeviceRayleighRitzResult


def _validate_state(operator: Any, state: DeviceSubspaceState) -> None:
    dimension = int(operator.shape[0])
    if tuple(operator.shape) != (dimension, dimension):
        raise ValueError("operator must be square")
    if state.operator_dimension != dimension:
        raise ValueError("saved subspace belongs to another operator dimension")
    if state.eigenvalues.shape != (state.working_states,):
        raise ValueError("saved eigenvalues do not match working_states")
    if state.vectors.shape != (dimension, state.working_states):
        raise ValueError("saved vectors do not match the operator")
    if not state.first_filter and state.filter_lower_bound is None:
        raise ValueError("continued state requires a filter lower bound")


def _next_filter_lower_bound(state: DeviceSubspaceState) -> float:
    largest = float(state.eigenvalues.max().item())
    if state.first_filter or state.filter_lower_bound is None:
        return max(largest, 0.0)
    return max(float(state.filter_lower_bound), largest)


def adapt_polynomial_degree(
    requested_degree: int,
    lower_bound: float,
    upper_bound: float,
) -> int:
    """Apply PARSEC's spectral-width lower limits to the filter degree."""

    interval = upper_bound - lower_bound
    if interval > 1400.0:
        return max(requested_degree, 15)
    if interval > 1200.0:
        return max(requested_degree, 11)
    if interval > 1000.0:
        return max(requested_degree, 9)
    if interval > 850.0:
        return max(requested_degree, 8)
    return requested_degree


def run_subspace_filter(
    operator: Any,
    state: DeviceSubspaceState,
    *,
    settings: SubspaceSettings = SubspaceSettings(),
    compute_residuals: bool = True,
    spectral_bound: LanczosBoundResult | None = None,
) -> DeviceSubspaceResult:
    """Perform one filter/orthogonalization/Ritz pass on the GPU.

    Residuals are optional because the later-SCF PARSEC policy does not use
    them to accept eigenpairs or to control SCF convergence.  Diagnostic
    callers retain the default; the production accelerated SCF disables the
    extra ``(H Q) C - (Q C) Lambda`` grid-by-state work.
    """

    _validate_state(operator, state)
    lower_bound = _next_filter_lower_bound(state)
    generator = np.random.default_rng(
        settings.random_seed + state.filters_completed
    )
    with device_stage(operator, "subspace_bound_seconds"):
        bound = (
            lanczos_upper_bound(
                operator, steps=settings.lanczos_steps, rng=generator
            )
            if spectral_bound is None
            else spectral_bound
        )
    upper_bound = float(bound.upper_bound)
    degree = adapt_polynomial_degree(
        settings.polynomial_degree, lower_bound, upper_bound
    )
    blocks = subspace_filter_blocks(
        state.working_states,
        settings.block_size,
        degree,
        settings.degree_delta,
    )
    with device_stage(operator, "subspace_filter_seconds"):
        filtered = subspace_filter(
            operator,
            state.vectors,
            degree=degree,
            degree_delta=settings.degree_delta,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            block_size=settings.block_size,
            reset_recurrence_per_block=settings.reset_recurrence_per_block,
        )
    use_generalized_ritz = bool(
        not state.generalized_ritz_failed
        and generalized_ritz_requested(*map(int, filtered.shape))
    )
    generalized_ritz_failed = state.generalized_ritz_failed
    if use_generalized_ritz:
        try:
            with device_stage(operator, "subspace_ritz_seconds"):
                ritz = generalized_rayleigh_ritz(
                    operator,
                    filtered,
                    workspace=state.ritz_workspace,
                    compute_residuals=bool(compute_residuals),
                )
        except GeneralizedRitzStabilityError:
            # An ill-conditioned overlap is not an SCF failure.  Householder
            # QR is the robust complete-basis fallback and remains selected
            # for subsequent iterations of this eigensolver state.
            generalized_ritz_failed = True
            with device_stage(operator, "subspace_orthogonalization_seconds"):
                basis = orthonormalize_complete_subspace(
                    filtered, rng=generator
                ).basis
            with device_stage(operator, "subspace_ritz_seconds"):
                ritz = rayleigh_ritz(
                    operator,
                    basis,
                    compute_residuals=bool(compute_residuals),
                )
    else:
        with device_stage(operator, "subspace_orthogonalization_seconds"):
            basis = orthonormalize_complete_subspace(filtered, rng=generator).basis
        with device_stage(operator, "subspace_ritz_seconds"):
            ritz = rayleigh_ritz(
                operator,
                basis,
                compute_residuals=bool(compute_residuals),
            )
    if ritz.eigenvalues.shape != (state.working_states,):
        raise RuntimeError("Rayleigh--Ritz changed the working state count")
    if ritz.wavefunctions.shape != (
        state.operator_dimension,
        state.working_states,
    ):
        raise RuntimeError("Rayleigh--Ritz changed the saved vector shape")

    next_state = DeviceSubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=state.working_states,
        eigenvalues=ritz.eigenvalues,
        vectors=ritz.wavefunctions,
        filter_lower_bound=lower_bound,
        first_filter=False,
        filters_completed=state.filters_completed + 1,
        ritz_workspace=ritz.workspace,
        generalized_ritz_failed=generalized_ritz_failed,
    )
    return DeviceSubspaceResult(
        eigenvalues=ritz.eigenvalues,
        vectors=ritz.wavefunctions,
        residual_norms=ritz.residual_norms,
        state=next_state,
        lanczos_bound=bound,
        polynomial_degree_used=degree,
        filter_blocks=blocks,
        rayleigh_ritz=ritz,
    )


__all__ = [
    "DeviceSubspaceResult",
    "DeviceSubspaceState",
    "SubspaceSettings",
    "adapt_polynomial_degree",
    "run_subspace_filter",
]
