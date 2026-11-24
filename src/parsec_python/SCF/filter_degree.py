"""The later-SCF Chebyshev degree of a loop near convergence, and its restoration where the loop stalls.

PARSEC lowers the polynomial degree of the later-SCF subspace filter by one per iteration, down to
``max(10, Chebyshev_Degree_Delta + 1)``, once the iteration is beyond ``Chebyshev_Degree_Decrease_After`` and the
weighted residual (the larger of the spin channels) is below ``Convergence_Criterion_Approach``
(:attr:`parsec_python.models.SCFSettings.approach_criterion`; ``parsec.F90`` l.1942-1952 of PARSEC 1.5).  The degree
is never raised again within the loop.  That saves Hamiltonian applications in a loop that goes on converging; in a
loop that does not, the filter one degree lower leaves the subspace of every step less converged, and the output
potential then depends on the subspaces of the steps before, not only on the input potential the mixer gave.
Anderson's secant step, which assumes the second, loses the slow directions of the residual (it needs residual
differences accurate to about ``alpha * delta`` of the residual along a direction whose fixed-point Jacobian is
``1 - delta``, a few per cent for such a direction).

``PARSEC_FILTER_DEGREE_STALL``:

* ``off`` (the default): the degree stays lowered, as in PARSEC 1.5.
* ``auto`` (opt-in): once the loop has lowered the degree, a loop that goes :data:`STALL_WINDOW` iterations without
  halving (:data:`STALL_PROGRESS`) the residual at the last reset (the anchor: the residual of the first drop, then of
  each halving), while the smallest residual since the drop is still in the decade below the approach criterion (at
  least :data:`STALL_DECADE` times it, where the degree is lowered), takes the starting degree back, keeps it to its
  end and prints a NOTE in its next iteration record (every loop that lowers the degree; no NOTE where PARSEC's
  spectral-width floor of the last SUBSPACE pass already kept the lowered degree at the restored one, so that the
  restoration changes nothing).  A loop that halves its residual within every window never meets the test and runs
  bit for bit as under ``off``; a loop that converges more slowly than that does meet it, and its path changes.

The degree is that of the SUBSPACE steps that follow the first solve of CHEBFF and CHEBDAV alike (and of ARPACK and
DIAGLA under ``Chebyshev_Filtering``).

Why ``auto`` is not the default (A100, first runs, ``auto`` against ``off`` side by side, 2026-11-23).  It rescues loops
that stall where the degree was lowered: the broken-symmetry O2 pair of the consistency matrix (``w_o2_broken``: an
open-shell singlet, pi*_x up and pi*_y down, whose orientation about the bond only the grid fixes; criterion 1e-6 Ry):
under ``off`` the matrix input and its starts with atom 1 moved by 1, 3, 5 and 7 x 1e-7 bohr converged 2 of 5 within
400 iterations (80, 312), under ``auto`` these and a start moved by 2 x 1e-7 6 of 6 (159, 209, 187, 80, 159, 177; the
start of 80 never stalls); at 1e-9 Ry not converged -> 229; at ``Expansion_Order: 12`` 400 -> 223.  It also shortens
some slow loops (Co4 at 300 K 251 -> 225 iterations, a bcc Fe2 k-point list at 1000 K 129 -> 110, SCF 245 -> 211 s,
the AuH spin-orbit cluster at 5e-3 Ry 64 -> 54).  But it lengthens others that converge without it: bcc Fe2 at 300 K
301 -> 345 iterations (SCF 21.8 -> 23.7 s, the same energy), and it costs the SCAN CH4 and NH3 runs of
``examples/0d_meta_gga``, which converge under neither value in 200 iterations, 7 to 9 % of SCF time.  Replayed on
the residual histories of 47,297 SCF loops of A100 runs, it would act in 396, of which 181 converged under ``off``.
Loops whose residual halves within the window (Mn2 antiferromagnetic on one process and on two MPI ranks, Ni32 at
1000 K, the O2 triplet, C7N, Si8, H2O, CH4) run bit for bit as under ``off``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

SWITCH = "PARSEC_FILTER_DEGREE_STALL"
#: The value without the variable.
DEFAULT = "off"
#: Iterations without halving the residual at the last reset (the anchor) after which the degree returns.
STALL_WINDOW = 20
#: The factor a residual must reach below the anchor to count as progress (and to become the new anchor).
STALL_PROGRESS = 0.5
#: The smallest weighted residual since the drop must be at least this times the approach criterion for the degree to
#: return (the decade below the criterion, where the degree is lowered).
STALL_DECADE = 0.1


def filter_degree_stall_setting() -> str:
    """The value of ``PARSEC_FILTER_DEGREE_STALL`` (``auto`` or ``off``); any other value is an error.  Read through
    :func:`~parsec_python.fortran_numerics.switch_value`: ``PARSEC_FORTRAN_NUMERICS=on`` gives PARSEC 1.5's ``off``
    where the variable is not set."""

    from ..fortran_numerics import switch_value

    value = switch_value(SWITCH, DEFAULT) or DEFAULT
    if value not in ("auto", "off"):
        raise ValueError(f"{SWITCH} must be auto or off, got {value!r}")
    return value


@dataclass
class LaterFilterDegree:
    """The later-SCF filter degree of one SCF loop (module docstring).

    ``degree`` is the degree of the next iteration and ``minimum`` the lowest the loop goes to;
    :meth:`after_iteration` updates the degree from the weighted residual of the iteration just finished.
    ``restore`` is whether the stall test runs (``PARSEC_FILTER_DEGREE_STALL=auto``); ``restored_at`` the iteration
    after which it gave the starting degree back (``None`` while it has not).  :meth:`take_note` hands the NOTE of a
    restoration to the next iteration record a loop builds, once.
    """

    degree: int
    minimum: int
    restore: bool = True
    initial: int = field(init=False)
    lowered: bool = field(default=False, init=False)
    restored_at: int | None = field(default=None, init=False)
    lowest: int | None = field(default=None, init=False)
    _anchor: float = field(default=math.inf, init=False)
    _best: float = field(default=math.inf, init=False)
    _since_anchor: int = field(default=0, init=False)
    _note_taken: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.degree = int(self.degree)
        self.initial = self.degree

    @classmethod
    def for_loop(cls, degree: int, minimum: int) -> "LaterFilterDegree":
        """The schedule of a loop that starts at ``degree`` and lowers it to ``minimum`` (the switch read here)."""

        return cls(degree=degree, minimum=minimum, restore=filter_degree_stall_setting() == "auto")

    def after_iteration(self, iteration: int, residual: float, approach: float, decrease_after: int) -> int:
        """The degree of the iteration after ``iteration``, whose weighted residual is ``residual`` (Ry), in a loop
        with the approach criterion ``approach`` (Ry) and ``Chebyshev_Degree_Decrease_After`` ``decrease_after``."""

        if self.restored_at is not None:
            return self.degree
        if self.restore and self.lowered:
            self._best = min(self._best, float(residual))
            if residual < STALL_PROGRESS * self._anchor:
                self._anchor = float(residual)
                self._since_anchor = 0
            else:
                self._since_anchor += 1
                if self._since_anchor >= STALL_WINDOW and self._best >= STALL_DECADE * approach:
                    self.restored_at = int(iteration)
                    self.lowest = self.degree
                    self.degree = max(self.degree, self.initial)
                    return self.degree
        if iteration > decrease_after and residual < approach and self.degree > self.minimum:
            if not self.lowered:
                self.lowered = True
                self._anchor = self._best = float(residual)
                self._since_anchor = 0
            self.degree -= 1
        return self.degree

    def take_note(self) -> tuple[str, ...]:
        """The NOTE of a restoration for the next iteration record, once; empty otherwise, and empty where PARSEC's
        spectral-width floor of the last SUBSPACE pass (``Eigensolvers.subspace.last_spectral_width_floor``) already
        kept the lowered degree at the restored one (the restoration then changes nothing)."""

        if self.restored_at is None or self._note_taken:
            return ()
        self._note_taken = True
        from ..Eigensolvers.subspace import last_spectral_width_floor

        floor = last_spectral_width_floor()
        lowered = max(int(self.lowest), floor)
        if lowered >= max(self.degree, floor):
            return ()
        return (
            f" NOTE: the later-SCF Chebyshev degree returns to {self.degree} (lowered to {lowered})"
            f" after iteration {self.restored_at}: {STALL_WINDOW} iterations without halving the weighted SRE"
            f" ({SWITCH}=auto).",
        )


__all__ = [
    "DEFAULT",
    "LaterFilterDegree",
    "STALL_DECADE",
    "STALL_PROGRESS",
    "STALL_WINDOW",
    "SWITCH",
    "filter_degree_stall_setting",
]
