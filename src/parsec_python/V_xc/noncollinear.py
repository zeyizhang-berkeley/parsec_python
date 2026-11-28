"""Non-collinear exchange-correlation from a collinear spin functional.

For a spinor density the local spin-density matrix is ``(n + m.sigma)/2`` with
charge density ``n`` and magnetization vector ``m(r)``.  Rotating at every
grid point to the local magnetization axis ``m_hat = m/|m|`` makes the matrix
diagonal with eigenvalues ``n_up,down = (n +/- |m|)/2``, so any collinear
spin-polarized functional (LSDA or PBE) applies unchanged,

``E_xc[n, m] = E_xc^collinear[(n + |m|)/2, (n - |m|)/2]``.

Its functional derivative has the form ``V_avg * 1 + B_xc . sigma`` with

``V_avg = (V_up + V_down)/2``,  ``B_xc = (V_up - V_down)/2 * m_hat``,

which follows from the chain rule through ``|m|`` (``d|m|/dm = m_hat``).  For
a gradient functional the gradients are those of the fields ``n_up`` and
``n_down`` built from ``|m|``, the usual local-rotation (``grad |m|``)
treatment.  ``B_xc`` is zero where ``|m|`` vanishes, which is the limit of the
collinear difference ``V_up - V_down``.

**The frames of a gradient functional** (``PARSEC_NONCOLLINEAR_GGA_FRAME``,
default ``auto``).  For a gradient functional ``(V_up - V_down)/2`` does not
vanish with ``|m|``: its gradient terms stay finite at a node of the
magnetization, while the direction ``m_hat`` there is that of whatever small
components ``m`` keeps (the transverse parts of a nearly collinear state, the
noise of the eigensolver), and ``|m|`` has a kink across a nodal surface.  The
field ``B_xc`` of the local frame is then discontinuous at the nodes and the SCF
stalls: a non-collinear O2 triplet with PBE at a potential residual of 1e-3 with
every mixer, a non-magnetic Au pair with spin-orbit coupling at 3e-8, while the
collinear spin driver converges the same O2 in 30 iterations.  ``auto`` decides
once per run, from the input (:func:`frame_decision`; kept by the moved
structures of a relaxation or dynamics, :func:`system_frame`):

* a gradient functional, no spin-orbit coupling, initial moments (the
  ``Initial_Spin_Polarization`` of each species times its ``Initial_NCL_Moment``)
  collinear along one axis ``u``, parallel or antiparallel: the **collinear
  frame**, the collinear functional of the signed projection ``sigma = m.u``,
  ``n_up,down = (n +/- m.u)/2``, its field ``(V_up - V_down)/2 u`` and potential
  ``(V_up + V_down)/2`` (the exact derivatives).  Without spin-orbit coupling a
  field along ``u`` keeps the spinors in the eigenstates of ``u.sigma``, so a
  collinear start stays collinear as in exact arithmetic (the transverse parts
  that a loose first solve leaves get no field); its result is that of the
  collinear spin driver.  A non-collinear state needs non-collinear initial
  moments;
* the same with spin-orbit coupling: the **axis frame**, the collinear
  functional of the signed polarization along the principal axis ``u`` of the
  magnetization (the eigenvector of the largest eigenvalue of ``sum_r m m^T``),

  ``sigma = a sqrt(1 + t^2 / (a^2 + eta^2))``, ``a = m.u``, ``t = |m - a u|``,
  ``eta = 1e-2 (n + n_core)``, ``n_up,down = (n +/- sigma)/2``,

  which is ``m.u`` for a magnetization along ``u`` (sign changes included),
  ``sign(a) |m|`` wherever ``|a| >> eta``, and smooth while the top eigenvalue of
  ``sum m m^T`` stays apart from the next; the field
  ``(V_up - V_down)/2 d sigma/dm`` (with the term of the dependence of ``u`` on
  ``m``) and the potential ``(V_up + V_down)/2 + (V_up - V_down)/2 d sigma/dn``
  are the exact derivatives of its energy, which is invariant under a global
  rotation of ``m``.  Spin-orbit coupling may cant the moments: where the
  magnetization leaves one axis the frame is wrong (``|sigma| < |m|`` within
  ``eta`` of the plane normal to ``u``) and jumps where the top eigenvalues cross,
  so a guard (:class:`NoncollinearFrame`) takes the local frame, with a NOTE,
  after ``GUARD_EVALUATIONS`` evaluations in a row of a magnetic state
  (``int |m| >= GUARD_MOMENT``) with the second eigenvalue above
  ``GUARD_RATIO`` of the first or more than ``GUARD_DEAD_BAND`` of ``int |m|``
  where ``|sigma| < |m|/2``;
* PBE or PW91 with non-collinear initial moments: the **multicollinear frame**
  (:mod:`parsec_python.V_xc.multicollinear`), the multicollinear functional of
  Pu et al. (Phys. Rev. Research 5, 013036 (2023)), an average of the collinear
  functional over projection axes, with a radial rule in place of its second
  derivatives and the local frame where ``|m|`` approaches ``n + n_core``: smooth
  in ``m`` and ``grad m`` at every point, the nodes of ``m`` among them, and the
  collinear functional in the collinear limit.  Where a CUDA device sums it
  (:func:`multicollinear_available`: CuPy with a device, not under
  ``PARSEC_CUPY_XC=off`` or ``PARSEC_DEVICE=cpu``); without one the host would
  take some 80 times the time of the local frame per evaluation, and ``auto``
  keeps the local frame, with a NOTE in parsec.out;
* every other run (a local functional, CA, PW92 or a libxc LDA, whose field
  vanishes with ``|m|``; no initial moment, as Quantum ESPRESSO's
  ``compute_ux``; the other gradient functionals with non-collinear initial
  moments): the local frame above.

``multicollinear``: the multicollinear frame for every run of PBE or PW91 (the
others as under ``auto``).  ``off`` (and ``PARSEC_FORTRAN_NUMERICS=on``): the
local frame for every run, as ``bxc.F90`` of PARSEC 1.5.  The collinear frame
is the signed frame of a fixed quantization axis that Quantum ESPRESSO takes
for a gradient functional when the starting moments are collinear
(``compute_rho`` with ``lsign``), with the field kept on that axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ca_lda import SpinPolarizedXCResult

_MAGNETIZATION_FLOOR = 1.0e-12

NONCOLLINEAR_GGA_FRAME_ENV = "PARSEC_NONCOLLINEAR_GGA_FRAME"
NONCOLLINEAR_GGA_FRAME_DEFAULT = "auto"
NONCOLLINEAR_GGA_FRAME_VALUES = ("auto", "multicollinear", "off")
#: the spin functionals without a density gradient: their field vanishes with ``|m|``, they keep the local frame
LOCAL_SPIN_FUNCTIONALS = frozenset(("ca", "pw92"))
#: the guard of the axis frame: a magnetic state (``int |m| >= GUARD_MOMENT`` Bohr magnetons) whose second eigenvalue
#: of ``sum m m^T`` is above ``GUARD_RATIO`` of the first, or with more than ``GUARD_DEAD_BAND`` of ``int |m|`` where
#: ``|sigma| < |m|/2``, in ``GUARD_EVALUATIONS`` evaluations in a row
GUARD_MOMENT = 0.05
GUARD_RATIO = 0.2
GUARD_DEAD_BAND = 0.1
GUARD_EVALUATIONS = 10
#: ``eta / (n + n_core)`` of the axis frame
AXIS_FRAME_SOFTNESS = 1.0e-2
#: two initial moments are collinear where ``|u x v| <= COLLINEAR_MOMENT_TOLERANCE |u| |v|``
COLLINEAR_MOMENT_TOLERANCE = 1.0e-6
#: ``auto`` takes the multicollinear frame for non-collinear initial moments of PBE and PW91 where a device sums it
#: (:func:`multicollinear_available`); the local frame without one
AUTO_MULTICOLLINEAR = True


@dataclass(frozen=True)
class NoncollinearXCResult:
    """``V_xc = potential * 1 + field . sigma`` and the XC energy (Rydberg).

    ``potential`` has shape ``(n_grid,)`` and ``field`` has shape
    ``(3, n_grid)``.
    """

    potential: np.ndarray
    field: np.ndarray
    energy_density: np.ndarray
    total_energy: float


def local_spin_densities(
    density: np.ndarray, magnetization: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n_up, n_down, |m|)`` along the local magnetization axis.

    ``|m|`` is clipped to ``n`` (a pure-state spinor density satisfies
    ``|m| <= n``; round-off can exceed it by ~1e-16).
    """

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    if magnetization.shape != (3, density.size):
        raise ValueError("magnetization must have shape (3, n_grid)")
    magnitude = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    magnitude = np.minimum(magnitude, np.maximum(density, 0.0))
    return 0.5 * (density + magnitude), 0.5 * (density - magnitude), magnitude


def noncollinear_gga_frame_setting() -> str:
    """``PARSEC_NONCOLLINEAR_GGA_FRAME``: ``auto`` (the default), ``multicollinear`` or ``off`` (the module
    docstring).  An unknown value, or one of ``PARSEC_NONCOLLINEAR_MC_POINTS``
    (:func:`parsec_python.V_xc.multicollinear.multicollinear_points_setting`), is an error."""

    from ..fortran_numerics import switch_value
    from .multicollinear import multicollinear_points_setting

    value = switch_value(NONCOLLINEAR_GGA_FRAME_ENV, NONCOLLINEAR_GGA_FRAME_DEFAULT)
    if value not in NONCOLLINEAR_GGA_FRAME_VALUES:
        raise ValueError(
            f"{NONCOLLINEAR_GGA_FRAME_ENV} must be one of {', '.join(NONCOLLINEAR_GGA_FRAME_VALUES)}, not {value!r}"
        )
    multicollinear_points_setting()
    return value


def is_local_functional(functional) -> bool:
    """Whether ``functional`` has no density gradient: CA, PW92, a libxc LDA (``libxc:lda_...``, not a hybrid)."""

    if functional in LOCAL_SPIN_FUNCTIONALS:
        return True
    if isinstance(functional, str) and functional.startswith("libxc:"):
        from .libxc import parse_libxc_spec

        try:
            spec = parse_libxc_spec(functional)
        except (ValueError, ImportError):
            return False
        return spec.family == "lda" and not spec.is_hybrid
    return False


def has_spin_orbit(problem) -> bool:
    """Whether a run (a ``SinglePointInput``) has spin-orbit coupling: ``SO_from_scratch`` or a species with
    ``SO_PSP``."""

    if getattr(problem.scf, "self_consistent_spin_orbit", False):
        return True
    return any(getattr(specification, "spin_orbit", False) for specification in problem.pseudopotentials.values())


def initial_moments(atoms, specifications) -> np.ndarray:
    """The initial moments, rows ``Initial_Spin_Polarization`` of the species times ``Initial_NCL_Moment`` of the atoms
    that have one."""

    rows = [
        float(getattr(specifications[atom.symbol], "initial_spin_polarization", 0.0) or 0.0)
        * np.asarray(atom.initial_moment, dtype=float)
        for atom in atoms
        if getattr(atom, "initial_moment", None) is not None
    ]
    return np.array(rows, dtype=float).reshape(-1, 3)


def initial_moment_axis(moments) -> tuple[bool, np.ndarray | None]:
    """Whether the initial moments (rows, zero rows ignored) are collinear, parallel or antiparallel, and their axis
    (the first nonzero row normalized); ``(False, None)`` without a nonzero moment."""

    rows = np.asarray(moments, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(rows, axis=1)
    nonzero = rows[norms > 0.0]
    if nonzero.shape[0] == 0:
        return False, None
    axis = nonzero[0] / np.linalg.norm(nonzero[0])
    collinear = all(
        np.linalg.norm(np.cross(axis, row)) <= COLLINEAR_MOMENT_TOLERANCE * np.linalg.norm(row) for row in nonzero[1:]
    )
    return bool(collinear), axis


FRAME_LOCAL = "local"
FRAME_COLLINEAR = "collinear"
FRAME_AXIS = "axis"
FRAME_MULTICOLLINEAR = "multicollinear"


class NoncollinearFrame:
    """The frame of the XC of one non-collinear run (the module docstring), decided once from its input
    (:func:`frame_decision`): ``mode`` ``local``, ``collinear`` (along the fixed ``axis``), ``axis`` (the principal
    axis of ``m``; ``axis`` where ``m`` is zero), with the guard of the axis frame, or ``multicollinear``
    (:mod:`parsec_python.V_xc.multicollinear`, which needs the system of the run).  ``left_axis_at`` is the evaluation
    at which the guard took the local frame (``None`` while it has not)."""

    def __init__(self, mode: str, axis: np.ndarray | None = None, setting: str = NONCOLLINEAR_GGA_FRAME_DEFAULT):
        if mode not in (FRAME_LOCAL, FRAME_COLLINEAR, FRAME_AXIS, FRAME_MULTICOLLINEAR):
            raise ValueError(f"unknown non-collinear frame {mode!r}")
        self.mode = mode
        self.axis = None if axis is None else np.asarray(axis, dtype=float)
        self.setting = setting
        self.evaluations = 0
        self.strikes = 0
        self.left_axis_at: int | None = None
        #: the local frame that ``auto`` keeps for non-collinear initial moments without a device for the
        #: multicollinear frame (:func:`frame_decision`)
        self.multicollinear_declined = False

    def __repr__(self) -> str:
        return f"NoncollinearFrame(mode={self.mode!r}, axis={self.axis!r}, left_axis_at={self.left_axis_at!r})"

    def guard_measures(
        self, density, magnetization, core_density=None, volume_element=None
    ) -> tuple[float, float, float]:
        """``(int |m|, lambda_2 / lambda_1, the share of int |m| where |sigma| < |m|/2)`` of the axis frame
        (``int |m|`` as the sum of ``|m|`` where ``volume_element`` is not given)."""

        density = np.asarray(density, dtype=float)
        magnetization = np.asarray(magnetization, dtype=float)
        magnitude = np.sqrt(np.sum(magnetization * magnetization, axis=0))
        total = float(np.sum(magnitude))
        moment = total * (1.0 if volume_element is None else float(volume_element))
        axis, values, _ = principal_axis(magnetization, self.axis)
        ratio = float(values[1] / values[2]) if values[2] > 0.0 else 0.0
        if not total > 0.0:
            return moment, ratio, 0.0
        scale = np.maximum(density if core_density is None else density + np.asarray(core_density, dtype=float), 0.0)
        a = axis @ magnetization
        t2 = np.maximum(magnitude * magnitude - a * a, 0.0)
        den = a * a + (AXIS_FRAME_SOFTNESS * scale) ** 2
        sigma = np.abs(a) * np.sqrt(1.0 + np.where(den > 0.0, t2 / np.where(den > 0.0, den, 1.0), 0.0))
        dead = sigma < 0.5 * magnitude
        return moment, ratio, float(np.sum(magnitude[dead])) / total

    def evaluate(
        self,
        density: np.ndarray,
        magnetization: np.ndarray,
        spin_functional: Callable[[np.ndarray, np.ndarray], SpinPolarizedXCResult],
        core_density: np.ndarray | None = None,
        volume_element: float | None = None,
        system=None,
    ) -> NoncollinearXCResult:
        """The XC of ``(n, m)`` in this frame (the guard of the axis frame counts every evaluation).  The
        multicollinear frame evaluates the functional of ``system`` (a prepared system: its grid, functional and core
        density) and needs it."""

        self.evaluations += 1
        if self.mode == FRAME_MULTICOLLINEAR:
            if system is None:
                raise ValueError("the multicollinear frame needs the system of the run")
            from .multicollinear import system_multicollinear_xc

            return system_multicollinear_xc(system, density, magnetization)
        if self.mode == FRAME_COLLINEAR:
            return collinear_frame_xc(density, magnetization, spin_functional, self.axis)
        if self.mode == FRAME_AXIS:
            moment, ratio, dead = self.guard_measures(density, magnetization, core_density, volume_element)
            if moment >= GUARD_MOMENT and (ratio > GUARD_RATIO or dead > GUARD_DEAD_BAND):
                self.strikes += 1
            else:
                self.strikes = 0
            if self.strikes < GUARD_EVALUATIONS:
                return axis_frame_xc(density, magnetization, spin_functional, self.axis, core_density)
            self.mode = FRAME_LOCAL
            self.left_axis_at = self.evaluations
        return noncollinear_xc(density, magnetization, spin_functional)


def frame_decision(
    functional, atoms, specifications, spin_orbit: bool, setting: str | None = None
) -> NoncollinearFrame:
    """The frame of a non-collinear run from its input (the module docstring): the local frame under ``off`` and for
    a local functional; the multicollinear frame for PBE and PW91 under ``multicollinear`` and, under ``auto``
    (:data:`AUTO_MULTICOLLINEAR`), for their non-collinear initial moments where a device sums it (else the local frame
    with ``multicollinear_declined``); the local frame without an initial moment
    and for the non-collinear moments of the other functionals; else the collinear frame along the axis of the initial
    moments without spin-orbit coupling and the axis frame with it.  ``atoms`` carry ``initial_moment`` (``None`` for
    an atom without one), ``specifications`` the species with ``initial_spin_polarization``."""

    from .multicollinear import MULTICOLLINEAR_FUNCTIONALS

    setting = noncollinear_gga_frame_setting() if setting is None else setting
    if setting == "off" or is_local_functional(functional):
        return NoncollinearFrame(FRAME_LOCAL, None, setting)
    if setting == "multicollinear" and functional in MULTICOLLINEAR_FUNCTIONALS:
        return NoncollinearFrame(FRAME_MULTICOLLINEAR, None, setting)
    collinear, axis = initial_moment_axis(initial_moments(atoms, specifications))
    if axis is not None and not collinear and AUTO_MULTICOLLINEAR and functional in MULTICOLLINEAR_FUNCTIONALS:
        if multicollinear_available():
            return NoncollinearFrame(FRAME_MULTICOLLINEAR, None, setting)
        # without a device the host would sum the rule at some 80 times the cost of the local frame
        frame = NoncollinearFrame(FRAME_LOCAL, None, setting)
        frame.multicollinear_declined = True
        return frame
    if axis is None or not collinear:
        return NoncollinearFrame(FRAME_LOCAL, None, setting)
    return NoncollinearFrame(FRAME_AXIS if spin_orbit else FRAME_COLLINEAR, axis, setting)


def multicollinear_available() -> bool:
    """Whether the multicollinear sum has a CUDA device in this process
    (:func:`parsec_python.acceleration.V_xc.multicollinear_xc.device_available`)."""

    from ..acceleration.V_xc.multicollinear_xc import device_available

    return device_available()


def problem_frame(problem, setting: str | None = None) -> NoncollinearFrame:
    """:func:`frame_decision` of a run's input (a ``SinglePointInput``)."""

    return frame_decision(
        problem.scf.xc_functional, problem.atoms, problem.pseudopotentials, has_spin_orbit(problem), setting
    )


def system_frame(system) -> NoncollinearFrame:
    """The frame of a prepared system's run: decided from its input when a non-collinear system is built and kept on
    it (``noncollinear_frame``), so that every copy (``replace``: the loosened copy of an upscaled relaxation SCF, the
    moved structures of ``update_geometry``, whose atoms the Fortran-derived routines rebuild without their initial
    moments) shares the same object; decided here for a system built without it."""

    frame = getattr(system, "noncollinear_frame", None)
    if frame is None:
        frame = problem_frame(system.input)
        try:
            object.__setattr__(system, "noncollinear_frame", frame)
        except AttributeError:
            pass
    return frame


def principal_axis(magnetization: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, values, vectors)``: the unit eigenvector ``u`` of the largest eigenvalue of ``sum_r m m^T`` (its largest
    component positive), with the ascending eigenvalues and the eigenvectors (columns); ``fallback`` where the
    magnetization is zero."""

    tensor = magnetization @ magnetization.T
    values, vectors = np.linalg.eigh(tensor)
    if not values[2] > 0.0:
        return np.asarray(fallback, dtype=float), values, vectors
    axis = vectors[:, 2]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis, values, vectors


def axis_frame_xc(
    density: np.ndarray,
    magnetization: np.ndarray,
    spin_functional: Callable[[np.ndarray, np.ndarray], SpinPolarizedXCResult],
    fallback_axis: np.ndarray,
    core_density: np.ndarray | None = None,
) -> NoncollinearXCResult:
    """Non-collinear XC of a gradient functional in the axis frame (the module docstring).

    ``spin_functional(n_up, n_down)`` adds its own share of ``core_density`` to each channel; ``core_density`` here
    only scales ``eta``, so that the returned potential is also the derivative in the core density (the core force).
    ``fallback_axis`` is the axis where the magnetization is zero."""

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    if magnetization.shape != (3, density.size):
        raise ValueError("magnetization must have shape (3, n_grid)")
    axis, values, vectors = principal_axis(magnetization, fallback_axis)
    positive = np.maximum(density, 0.0)
    scale = positive if core_density is None else np.maximum(density + np.asarray(core_density, dtype=float), 0.0)
    a = axis @ magnetization
    perp = magnetization - np.outer(axis, a)
    t2 = np.sum(perp * perp, axis=0)
    eta2 = (AXIS_FRAME_SOFTNESS * scale) ** 2
    den = a * a + eta2
    active = den > 0.0
    safe = np.where(active, den, 1.0)
    ratio = np.where(active & (t2 > 0.0), t2 / safe, 0.0)
    root = np.sqrt(1.0 + ratio)
    sigma = a * root
    # d sigma / da, d sigma / d(t^2), d sigma / d(eta^2)
    d_a = np.where(active, root - a * a * t2 / (safe * safe * root), 1.0)
    d_t2 = np.where(active, a / (2.0 * root * safe), 0.0)
    d_eta2 = np.where(active, -a * t2 / (2.0 * root * safe * safe), 0.0)
    # |sigma| <= |m| <= n; a point past n by round-off takes sigma = +-n, whose derivatives are 0 in m and +-1 in n
    clipped = np.abs(sigma) > positive
    sign = np.where(a < 0.0, -1.0, 1.0)
    sigma = np.where(clipped, sign * positive, sigma)

    collinear = spin_functional(0.5 * (density + sigma), 0.5 * (density - sigma))
    average = 0.5 * (collinear.potential_up + collinear.potential_down)
    delta = 0.5 * (collinear.potential_up - collinear.potential_down)
    d_a = np.where(clipped, 0.0, d_a)
    d_t2 = np.where(clipped, 0.0, d_t2)
    d_n = np.where(clipped, np.where(density > 0.0, sign, 0.0), d_eta2 * 2.0 * AXIS_FRAME_SOFTNESS ** 2 * scale)
    field = delta[None, :] * (np.outer(axis, d_a) + 2.0 * d_t2[None, :] * perp)
    if values[2] > 0.0:
        # the dependence of u on m: d u = sum_k v_k v_k^T dT u / (lambda_u - lambda_k), dT = dm m^T + m dm^T
        coupling = magnetization @ (delta * (d_a - 2.0 * a * d_t2))
        for k in (0, 1):
            gap = values[2] - values[k]
            if gap > 0.0:
                weight = float(coupling @ vectors[:, k]) / gap
                if weight != 0.0:
                    field = field + weight * (
                        np.outer(vectors[:, k], a) + np.outer(axis, vectors[:, k] @ magnetization)
                    )
    return NoncollinearXCResult(
        potential=average + delta * d_n,
        field=field,
        energy_density=collinear.energy_density,
        total_energy=collinear.total_energy,
    )


def collinear_frame_xc(
    density: np.ndarray,
    magnetization: np.ndarray,
    spin_functional: Callable[[np.ndarray, np.ndarray], SpinPolarizedXCResult],
    axis: np.ndarray,
) -> NoncollinearXCResult:
    """Non-collinear XC in the collinear frame along the fixed unit vector ``axis`` (the module docstring): the
    collinear functional of ``sigma = m.axis`` (clipped to ``|sigma| <= n``), the field ``(V_up - V_down)/2 axis``."""

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    if magnetization.shape != (3, density.size):
        raise ValueError("magnetization must have shape (3, n_grid)")
    axis = np.asarray(axis, dtype=float)
    positive = np.maximum(density, 0.0)
    a = axis @ magnetization
    sigma = np.clip(a, -positive, positive)
    collinear = spin_functional(0.5 * (density + sigma), 0.5 * (density - sigma))
    delta = 0.5 * (collinear.potential_up - collinear.potential_down)
    # a point past n by round-off: sigma = +-n, whose derivatives are 0 in m and +-1 in n
    clipped = sigma != a
    average = 0.5 * (collinear.potential_up + collinear.potential_down)
    if np.any(clipped):
        average = average + np.where(clipped & (density > 0.0), np.sign(a) * delta, 0.0)
        delta = np.where(clipped, 0.0, delta)
    return NoncollinearXCResult(
        potential=average,
        field=np.outer(axis, delta),
        energy_density=collinear.energy_density,
        total_energy=collinear.total_energy,
    )


def noncollinear_xc(
    density: np.ndarray,
    magnetization: np.ndarray,
    spin_functional: Callable[[np.ndarray, np.ndarray], SpinPolarizedXCResult],
) -> NoncollinearXCResult:
    """Evaluate non-collinear XC with a collinear ``spin_functional(n_up, n_down)`` in the local frame (the frames
    of a gradient functional: :class:`NoncollinearFrame`)."""

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    n_up, n_down, magnitude = local_spin_densities(density, magnetization)
    collinear = spin_functional(n_up, n_down)

    average = 0.5 * (collinear.potential_up + collinear.potential_down)
    delta = 0.5 * (collinear.potential_up - collinear.potential_down)
    raw_magnitude = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    usable = raw_magnitude > _MAGNETIZATION_FLOOR
    direction = np.zeros_like(magnetization)
    direction[:, usable] = magnetization[:, usable] / raw_magnitude[usable]
    return NoncollinearXCResult(
        potential=average,
        field=delta[None, :] * direction,
        energy_density=collinear.energy_density,
        total_energy=collinear.total_energy,
    )


__all__ = [
    "AXIS_FRAME_SOFTNESS",
    "NONCOLLINEAR_GGA_FRAME_ENV",
    "NoncollinearFrame",
    "NoncollinearXCResult",
    "axis_frame_xc",
    "collinear_frame_xc",
    "frame_decision",
    "has_spin_orbit",
    "initial_moment_axis",
    "initial_moments",
    "is_local_functional",
    "local_spin_densities",
    "noncollinear_gga_frame_setting",
    "noncollinear_xc",
    "principal_axis",
    "problem_frame",
    "system_frame",
]
