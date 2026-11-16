"""Opt-in reconstruction of a checkpoint's missing eigensubspace.

Charge/potential checkpoints do not contain wavefunctions.  A fixed number
of CHEBFF cycles need not rebuild the subspace accurately enough to mix its
density into an already good potential.  This policy holds that potential
fixed and requires stable requested Ritz levels on two successive passes.
It is a level-stability gate, not a bound on the eigenvector residual or a
replacement for the full SCF convergence test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
import math
import os
from time import perf_counter

import numpy as np

from .. import diagnostics as debug


class RestartRefinementError(RuntimeError):
    """Stop before density construction/mixing when reconstruction fails."""


@dataclass(frozen=True)
class RestartRefinementSettings:
    tolerance: float
    max_passes: int = 100
    stable_passes: int = 2
    lower_bound_rule: str = "inherit"

    def __post_init__(self):
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("restart level tolerance must be finite and positive")
        if (not isinstance(self.max_passes, int) or not isinstance(self.stable_passes, int)
                or not 1 <= self.stable_passes <= self.max_passes):
            raise ValueError("restart refinement needs 1 <= stable_passes <= max_passes")
        if self.lower_bound_rule not in ("inherit", "smart"):
            raise ValueError("restart refinement lower bound must be inherit or smart")


def from_environment(eigensolver_settings):
    """Explicit experimental switch; tolerance is the input Diag_Tolerance.

    All ranks must receive identical environment settings.  No dependence
    on PARSEC_DEBUG: logging must not decide scientific behavior.
    """
    mode = os.environ.get("PARSEC_RESTART_REFINE", "off").strip().lower()
    if mode == "off":
        return None
    if mode != "levels":
        raise ValueError("PARSEC_RESTART_REFINE must be off or levels")
    return RestartRefinementSettings(
        float(eigensolver_settings.tolerance),
        max_passes=int(os.environ.get("PARSEC_RESTART_REFINE_MAX_PASSES", "100")),
        lower_bound_rule=os.environ.get("PARSEC_RESTART_REFINE_LOWER_BOUND", "inherit").strip().lower(),
    )


def refinement_settings(settings, policy):
    """An optional bound policy for fixed-potential reconstruction passes.

    Reuse the existing PARSEC smart bound, without changing the first solve,
    polynomial degree, precision, acceptance tolerance or stability fallback.
    No array storage is copied. The caller's settings remain unchanged; the
    reconstructed subspace (including its bound history) is retained.
    """
    if policy.lower_bound_rule == "inherit":
        return settings
    return replace(settings, subspace=replace(settings.subspace,
                                             lower_bound_rule=policy.lower_bound_rule))


def _levels(values):
    # Snapshot the small host spectrum: some backends reuse its storage.
    blocks = tuple(np.array(v, dtype=np.float64, copy=True) for v in values)
    if not blocks or any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in blocks):
        raise RestartRefinementError("Restart level refinement failed: invalid/nonfinite spectrum")
    return blocks


def refine(step, initial_levels, policy, *, label):
    """Call a fixed-Hamiltonian step, retaining O(states) host data only.

    ``step`` owns the subspace and returns one spectrum per task/channel.
    It must release obsolete orbital wrappers before allocating a successor.
    No density, mixer, checkpoint callback or potential update occurs here.
    """
    previous = _levels(initial_levels)
    started = perf_counter()
    stable = 0
    for number in range(1, policy.max_passes + 1):
        debug.emit("restart_level_refinement_start", label=label, passes=number,
                   lower_bound_rule=policy.lower_bound_rule, tolerance_ry=policy.tolerance)
        current = _levels(step())
        if len(current) != len(previous) or any(a.shape != b.shape for a, b in zip(current, previous)):
            raise RestartRefinementError("Restart level refinement failed: state inventory changed")
        change = max(float(np.max(np.abs(a - b))) for a, b in zip(current, previous))
        stable = stable + 1 if change <= policy.tolerance else 0
        report = dict(label=label, passes=number, maximum_level_change_ry=change,
                      tolerance_ry=policy.tolerance, stable_passes=stable,
                      required_stable_passes=policy.stable_passes,
                      lower_bound_rule=policy.lower_bound_rule,
                      seconds=perf_counter() - started,
                      passed=stable >= policy.stable_passes)
        debug.emit("restart_level_refinement", **report)
        previous = current
        if report["passed"]:
            return report
    raise RestartRefinementError(
        f"Restart level refinement failed after {policy.max_passes} passes: "
        f"level change {change:.8g} Ry, tolerance {policy.tolerance:.8g} Ry; "
        "no new density or mixed checkpoint was accepted"
    )


def guarded_solver(solve, policy):
    """Guard a scalar/sector solve only when no saved eigenstate exists."""
    if policy is None:
        return solve

    @wraps(solve)
    def guarded(operator, count, *, settings, state=None, **kwargs):
        solution = solve(operator, count, settings=settings, state=state, **kwargs)
        if state is not None:
            return solution
        reconstruction_settings = refinement_settings(settings, policy)

        def step():
            nonlocal solution
            saved = solution.state
            drop = getattr(solution.vectors, "release_intermediate_storage", None)
            if callable(drop):
                drop()
            # Do not retain the result wrapper across the next allocation.
            solution = None
            solution = solve(operator, count, settings=reconstruction_settings, state=saved, **kwargs)
            return (solution.eigenvalues,)

        refine(step, (solution.eigenvalues,), policy, label="scalar_or_sectors")
        return solution

    return guarded


def refine_kpoints(solver, potentials, degree, tolerance, eigenvalues, policy):
    """Same gate across every k-point/spin task, with unchanged potentials."""
    current = eigenvalues

    def step():
        nonlocal current
        current = solver.solve(potentials, degree, tolerance)
        return current

    if policy.lower_bound_rule == "inherit":
        refine(step, current, policy, label="kpoints")
        return current

    # Every built-in serial/device/MPI k-point solver builds per-call settings
    # from its KPointJob. Apply the same policy to every task/spin channel and
    # restore the factory even when a reconstruction gate fails. Each MPI
    # rank enters this scope with the same policy; no basis is gathered.
    job = getattr(solver, "job", None)
    if job is None or not callable(getattr(job, "eigval_settings", None)):
        raise TypeError("restart filter override requires a solver with a KPointJob")
    original = job.eigval_settings

    def configured(*args, **kwargs):
        return refinement_settings(original(*args, **kwargs), policy)

    job.eigval_settings = configured
    try:
        refine(step, current, policy, label="kpoints")
    finally:
        job.eigval_settings = original
    return current
