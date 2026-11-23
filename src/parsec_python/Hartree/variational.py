"""The Hartree potential of a cluster as the derivative of its Hartree energy.

The Poisson problem of a cluster with multipole boundary values solves, on
the active points ``I`` with the missing stencil points ``B``,

``A V = 8 pi rho - A_IB M rho + t``,

where ``A = A_II`` is the interior negative Laplacian, ``M`` maps the
density to its order-``L`` multipole series at the exterior points and ``t``
are the rows of the atomic tail (:mod:`parsec_python.Hartree.boundary`).  So
``V_H = K rho + w`` with ``K = A^-1 (8 pi - A_IB M)`` and ``w = A^-1 t``.
The Hartree energy is ``E_H = (h^3/2) rho.V_H``.  ``K`` is not symmetric:
the multipole series is exact only for density well inside the sphere, and
the density of the outer layer is summed into moments that the boundary does
not return to it.  The tail depends on the positions and, through the
charges ``q_a = Z_a N_e / sum Z``, on the electron count.  Hence ``V_H`` is
not the derivative of ``E_H``, the SCF does not make ``E_H`` stationary, and
the Hellmann-Feynman forces, which assume it does, miss a term of the
density response.  An Au2 cluster in a sphere of 8 bohr was 4.7e-5 Ry/bohr
from the slope of its energy.

With the tail taken proportional to the electron count (as it is built),
the derivative of the same energy is

``dE_H/drho = K_s rho + w/2 + (rho.w)/(2 sum rho)``,   ``K_s = (K + K^T)/2``,

and, with ``u = A^-1 rho`` (zero boundary values) and ``K^T rho = 8 pi u -
M^T A_IB^T u``,

``V' = (V_H + 8 pi u - M^T A_IB^T u)/2 + (u.t)/(2 sum rho)``.

Taken as the Hartree potential of the SCF, ``V'`` keeps the energy
functional: ``(h^3/2) rho.V' = (h^3/2) rho.V_H`` at every density (the
asymmetric part is orthogonal to ``rho``, the constant carries the missing
quarter of the tail), so the usual bookkeeping of the total energy is
unchanged, and the self-consistent density makes it stationary.  The forces
then need only the explicit dependence of the tail on the positions,

``F_a = (h^3/2) d/dR_a [y.C_L(R)]``,   ``y = A_IB^T u``,

with ``C_L(P) = 2 sum_a q_a/|P - R_a| - M_L[q](P)`` at the exterior points.

What this buys is consistency, not accuracy: the forces become the exact
slope of the energy functional of the multipole boundary, which is what a
finite-difference check of the forces and the stopping test of a relaxation
compare with.  That functional is itself an approximation of the one with
the boundary values summed directly (``Full_Hartree``), and the slopes of the
two differ by up to 2e-4 Ry/bohr in small spheres; the forces of the Poisson
potential lie closer to the slope of the direct functional than those of
``V'`` on every cluster measured (Au2, Ag2H2, Ge2H2, Si2H6, Na4).  The
potential that would make the forces the slope of the direct functional is
the transpose of the direct Coulomb sums of the boundary, not of the
multipole series; it is not built here.

Each SCF step solves one more Poisson problem (``u``, from the extrapolation
of the two before, to the accuracy of the solve of ``V_H``) and evaluates the
transposed multipole map, a solid-harmonic series, at the points of the
fields.  ``PARSEC_HARTREE_VARIATIONAL`` selects it
(:func:`hartree_variational_setting`); it is off by default.  The change is
negligible where the density at the sphere is (the forces of H2O at the
radius of the default rule moved by 1e-9 Ry/bohr, of Na2 in a sphere of 12
bohr by 3e-8), so ``auto`` takes it only where the first solve of a structure
estimates a force change of :data:`VARIATIONAL_FORCE_THRESHOLD` or more.  The
estimate, the self-consistency force term of ``forcloc.f90`` with ``V' - V_H``
as the residual plus the force of the tail, was 0.07 to 50 times the change
it estimates on the clusters measured.

``V'`` is not the electrostatic potential of the density (near the sphere it
differs from ``V_H`` by up to 5e-2 Ry), so the results keep ``V_H`` as their
Hartree potential (:func:`electrostatic_hartree`): the archive, the cube
files and the files of a frozen-density embedding hold the potential of the
Poisson solve, the effective potentials of the SCF the one it ran with.

A slab whose forces are computed takes the same potential under ``on`` and
``auto`` with the sheet sums of its boundary values in place of the multipole
series (:class:`ConfinedVariationalHartree`; no tail, no estimate); a wire
keeps the Poisson solution.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
import os
from time import perf_counter

import numpy as np

from ..Laplacian import second_derivative_coefficients


HARTREE_VARIATIONAL_SWITCH = "PARSEC_HARTREE_VARIATIONAL"
HARTREE_VARIATIONAL_VALUES = ("auto", "on", "off")
#: ``auto``: the force change (Ry/bohr), estimated at the first density of a
#: structure, from which a run takes V' (:meth:`VariationalHartree._decide`).
VARIATIONAL_FORCE_THRESHOLD = 3.0e-7


def hartree_variational_setting() -> str:
    """``PARSEC_HARTREE_VARIATIONAL``: ``off`` (the default), ``auto`` or ``on``.

    ``off``: the potential of the Poisson solve, as PARSEC.  ``auto``: the
    Hartree potential of :mod:`this module <parsec_python.Hartree.variational>`
    in a cluster calculation with a multipole boundary whose forces are
    computed (a relaxation, molecular dynamics, manual movements without
    ``Skip_Force: true``, ``Calculate_Force``) where the first solve of a
    structure estimates that it moves the forces by at least
    :data:`VARIATIONAL_FORCE_THRESHOLD` (:meth:`VariationalHartree._decide`);
    below it, and with ``Hartree_Boundary_Tolerance: off`` (PARSEC's boundary
    values, for a comparison with Fortran PARSEC), the run keeps the potential
    of the Poisson solve, bit for bit that of ``off``.  ``on``: in every such
    cluster calculation, without the estimate.  A slab whose forces are
    computed takes the potential of its sheet sums under ``on`` and ``auto``
    alike, without an estimate (:class:`ConfinedVariationalHartree`); a wire
    keeps the Poisson solution.  An unknown value is refused.
    """

    value = os.environ.get(HARTREE_VARIATIONAL_SWITCH, "off").strip().lower() or "off"
    if value not in HARTREE_VARIATIONAL_VALUES:
        raise ValueError(
            f"{HARTREE_VARIATIONAL_SWITCH}={value!r}: expected one of "
            + ", ".join(HARTREE_VARIATIONAL_VALUES)
        )
    return value


def variational_hartree_applies(settings, grid_settings, *, forces_wanted: bool, setting: str | None = None) -> bool:
    """Whether a cluster calculation takes the variational Hartree potential.

    A spherical domain with multipole boundary values only (a box, another
    domain shape or ``Full_Hartree`` sums the boundary directly); under
    ``auto`` only where the forces are computed and the boundary is the one
    of this code: ``Hartree_Boundary_Tolerance: off``, the request for
    PARSEC's boundary values (a comparison with Fortran PARSEC), keeps
    PARSEC's potential as well.
    """

    setting = hartree_variational_setting() if setting is None else setting
    if setting == "off":
        return False
    if getattr(grid_settings, "domain_shape", None) != "sphere":
        return False
    if getattr(settings, "boundary_method", "auto") not in ("auto", "multipole"):
        return False
    if setting == "on":
        return True
    return bool(forces_wanted) and getattr(settings, "boundary_tolerance", None) is not None


def _normalization(degree: int, order: int) -> float:
    log_ratio = math.lgamma(degree - order + 1) - math.lgamma(degree + order + 1)
    return math.sqrt((2 * degree + 1) * math.exp(log_ratio) / (4.0 * math.pi))


def _solid_recurrence(x, y, z, square, order: int, xp=np):
    """Yield ``(m, C_m, S_m, l, q_l^m)`` for ``0 <= m <= l <= order``, ``m`` outer.

    ``r^l P_l^m(cos theta) e^(i m phi) = q_l^m (x + i y)^m`` with the ``P_l^m``
    of :func:`parsec_python.Hartree.harmonics._positive_m_harmonic_rows`
    (Condon-Shortley phase): ``q_m^m = (-1)^m
    (2m-1)!!``, ``q_(m+1)^m = (2m+1) z q_m^m`` and ``(l-m) q_l^m = (2l-1) z
    q_(l-1)^m - (l+m-1) r^2 q_(l-2)^m``, polynomials of the coordinates
    without roots or divisions by ``r``.  ``C_m + i S_m = (x + i y)^m``.
    """

    cosine = xp.ones_like(x)
    sine = xp.zeros_like(x)
    diagonal = 1.0
    for order_m in range(order + 1):
        if order_m:
            cosine, sine = cosine * x - sine * y, cosine * y + sine * x
            diagonal *= -(2 * order_m - 1)
        older = None
        previous = xp.full_like(x, diagonal)
        yield order_m, cosine, sine, order_m, previous
        for degree in range(order_m + 1, order + 1):
            if older is None:
                current = (2 * order_m + 1) * z * previous
            else:
                current = ((2 * degree - 1) * z * previous - (degree + order_m - 1) * square * older) / (degree - order_m)
            yield order_m, cosine, sine, degree, current
            older, previous = previous, current


def point_charge_series(points, positions, charges, order: int, xp=np):
    """``M_L[q](P)``, the order-``L`` multipole series of point charges, at ``points`` (Ry).

    That of :func:`parsec_python.Hartree.boundary.point_charge_multipoles`
    (``2 sum_lm 4 pi/(2l+1) Q_lm Y_lm(P)/r^(l+1)`` with ``Q_lm = sum_a q_a
    r_a^l conj(Y_lm(R_a))``) by the recurrences of :func:`_solid_recurrence`.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    moments = np.zeros((order + 1, order + 1), dtype=np.complex128)
    square = np.einsum("ij,ij->i", positions, positions)
    for order_m, cosine, sine, degree, q in _solid_recurrence(
        positions[:, 0], positions[:, 1], positions[:, 2], square, order
    ):
        weighted = charges * q
        moments[degree, order_m] = _normalization(degree, order_m) * complex(
            np.dot(weighted, cosine), -np.dot(weighted, sine)
        )
    points = xp.asarray(points, dtype=xp.float64)
    x, y, z = (xp.ascontiguousarray(points[:, axis]) for axis in range(3))
    square = x * x + y * y + z * z
    inverse_square = 1.0 / square
    powers = [1.0 / xp.sqrt(square)]
    for _ in range(order):
        powers.append(powers[-1] * inverse_square)
    result = xp.zeros(points.shape[0], dtype=xp.float64)
    real = imaginary = None
    current = -1
    for order_m, cosine, sine, degree, q in _solid_recurrence(x, y, z, square, order, xp):
        if order_m != current:
            if real is not None:
                result += previous_cosine * real - previous_sine * imaginary
            real = xp.zeros_like(result)
            imaginary = xp.zeros_like(result)
            current = order_m
        previous_cosine, previous_sine = cosine, sine
        moment = complex(moments[degree, order_m])
        weight = (1.0 if order_m == 0 else 2.0) * 8.0 * math.pi / (2 * degree + 1) * _normalization(degree, order_m)
        scaled = q * powers[degree]
        if moment.real:
            real += (weight * moment.real) * scaled
        if moment.imag:
            imaginary += (weight * moment.imag) * scaled
    if real is not None:
        result += previous_cosine * real - previous_sine * imaginary
    return result


def coulomb_sums(points, positions, charges, xp=np, *, weights=None, chunk_pairs: int = 8_000_000):
    """Pair sums between point charges and ``points``, in blocks of atoms, with the array module ``xp``.

    Without ``weights``: ``2 sum_a q_a/|P - R_a|`` at each point (Ry).  With
    ``weights`` ``w`` (one per point): ``2 q_a sum_P w_P (P - R_a)/|P -
    R_a|^3``, shape ``(atoms, 3)``, the gradient of ``sum_P w_P 2 q_a/|P -
    R_a|`` with respect to ``R_a``.  A block holds at most ``chunk_pairs``
    pairs.
    """

    points = xp.asarray(points, dtype=xp.float64)
    positions = xp.asarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3))
    charges = xp.asarray(np.asarray(charges, dtype=np.float64))
    px, py, pz = (xp.ascontiguousarray(points[:, axis])[:, None] for axis in range(3))
    count, atoms = int(points.shape[0]), int(positions.shape[0])
    block = max(1, int(chunk_pairs) // max(1, count))
    if weights is None:
        result = xp.zeros(count, dtype=xp.float64)
    else:
        weights = xp.asarray(weights, dtype=xp.float64)
        result = xp.zeros((atoms, 3), dtype=xp.float64)
    for start in range(0, atoms, block):
        stop = min(atoms, start + block)
        dx = px - positions[None, start:stop, 0]
        dy = py - positions[None, start:stop, 1]
        dz = pz - positions[None, start:stop, 2]
        inverse = 1.0 / xp.sqrt(dx * dx + dy * dy + dz * dz)
        if weights is None:
            result += inverse @ charges[start:stop]
            continue
        cubed = inverse * inverse * inverse
        result[start:stop, 0] = weights @ (dx * cubed)
        result[start:stop, 1] = weights @ (dy * cubed)
        result[start:stop, 2] = weights @ (dz * cubed)
    if weights is None:
        return 2.0 * result
    return 2.0 * charges[:, None] * result


def _array_module(device):
    """``(xp, context)``: NumPy on the host for ``device`` None, else CuPy on that CUDA device."""

    if device is None:
        return np, nullcontext()
    from ..acceleration.backends.cupy import require_cupy

    cp, _ = require_cupy()
    return cp, cp.cuda.Device(int(device))


def _host(values) -> np.ndarray:
    return values.get() if hasattr(values, "get") else np.asarray(values)


@dataclass(frozen=True)
class BoundaryPairs:
    """The block ``A_IB`` of the negative Laplacian, one entry per pair.

    ``rows`` are the active rows with a missing stencil neighbour, once per
    missing neighbour; ``coefficients`` the entries ``-c_|j| / h^2`` that
    :func:`parsec_python.Laplacian.apply_negative_laplacian_boundary`
    subtracts; ``exterior_index`` the exterior point of each pair among the
    unique points ``exterior`` (bohr), whose potential the boundary values
    give.
    """

    rows: np.ndarray
    coefficients: np.ndarray
    exterior_index: np.ndarray
    exterior: np.ndarray

    @property
    def exterior_count(self) -> int:
        return int(self.exterior.shape[0])


def boundary_pairs(grid) -> BoundaryPairs:
    """Walk the stencil of ``grid`` as the boundary correction of the Poisson right-hand side does.

    In a sphere of radius ``R`` only the rows beyond ``R - (M + 1) h`` can
    miss a neighbour (``M`` the half width of the stencil): every point of
    the sphere is active, so the walk visits those rows alone.
    """

    coefficients = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    inverse_square = 1.0 / grid.spacing**2
    if getattr(grid.settings, "domain_shape", None) == "sphere":
        distance = np.sqrt(np.einsum("ij,ij->i", grid.coordinates, grid.coordinates))
        candidates = np.flatnonzero(distance > float(grid.settings.radius) - (width + 1) * grid.spacing)
    else:
        candidates = np.arange(grid.size)
    base = grid.integer_coordinates[candidates]
    rows, values, points = [], [], []
    for axis in range(3):
        for shell in range(-width, width + 1):
            if shell == 0:
                continue
            displaced = base.copy()
            displaced[:, axis] += shell
            missing = grid.rows_for_integer_coordinates(displaced) < 0
            if not np.any(missing):
                continue
            rows.append(candidates[missing])
            values.append(np.full(int(missing.sum()), -coefficients[width + abs(shell)] * inverse_square))
            points.append(displaced[missing])
    if not rows:
        empty = np.zeros(0, dtype=np.int64)
        return BoundaryPairs(empty, np.zeros(0), empty, np.zeros((0, 3)))
    integer_points = np.concatenate(points).astype(np.int64, copy=False)
    # unique points by one integer key per point (np.unique of rows sorts them lexicographically, far slower)
    low = integer_points.min(axis=0)
    span = integer_points.max(axis=0) - low + 1
    key = ((integer_points[:, 0] - low[0]) * span[1] + (integer_points[:, 1] - low[1])) * span[2] + (
        integer_points[:, 2] - low[2]
    )
    unique_key, inverse = np.unique(key, return_inverse=True)
    unique = np.stack(
        [
            unique_key // (span[1] * span[2]) + low[0],
            (unique_key // span[2]) % span[1] + low[1],
            unique_key % span[2] + low[2],
        ],
        axis=1,
    )
    return BoundaryPairs(
        rows=np.ascontiguousarray(np.concatenate(rows), dtype=np.int64),
        coefficients=np.ascontiguousarray(np.concatenate(values), dtype=np.float64),
        exterior_index=np.ascontiguousarray(np.asarray(inverse).ravel(), dtype=np.int64),
        exterior=np.ascontiguousarray(grid.physical_coordinates(unique), dtype=np.float64),
    )


class FieldLayout:
    """How a scalar field of the SCF is held: the full grid, or one physical value per orbit of ``reduction``."""

    def __init__(self, grid, reduction=None):
        self.grid = grid
        self.reduction = reduction

    @classmethod
    def of(cls, grid, field):
        reduction = getattr(field, "reduction", None)
        return cls(grid, reduction if hasattr(field, "values") else None)

    def values(self, field) -> np.ndarray:
        values = field.values if self.reduction is not None and hasattr(field, "values") else field
        return np.asarray(values, dtype=np.float64)

    def wrap(self, values, like):
        if self.reduction is None:
            return values
        return type(like)(self.reduction, values)

    def at_rows(self, values, rows) -> np.ndarray:
        """The physical values of a field at full-grid ``rows``."""

        if self.reduction is None:
            return values[rows]
        return values[self.reduction.full_to_wedge[rows]]

    def total(self, values) -> float:
        """The sum of a field over the full grid."""

        if self.reduction is None:
            return float(np.sum(values))
        return float(np.dot(self.reduction.multiplicities, values))

    def points(self) -> np.ndarray:
        """The grid point that holds each value."""

        if self.reduction is None:
            return self.grid.coordinates
        return self.grid.coordinates[self.reduction.representative_rows]

    def convert(self, values, target: "FieldLayout") -> np.ndarray:
        """The same invariant field in the layout ``target``."""

        if target.reduction is self.reduction:
            return values
        full = values if self.reduction is None else values[self.reduction.full_to_wedge]
        if target.reduction is None:
            return full
        # the orbit average, as the projection onto invariant fields does
        reduction = target.reduction
        sums = np.bincount(reduction.full_to_wedge, weights=full, minlength=reduction.wedge_size)
        return sums / np.asarray(reduction.multiplicities, dtype=np.float64)

    @property
    def size(self) -> int:
        return self.grid.size if self.reduction is None else self.reduction.wedge_size


class HartreeAdjoint:
    """``M^T A_IB^T`` of the multipole boundary of one grid, and the derivative of the atomic tail.

    ``device``: the CUDA device of the sums over atoms and exterior points of
    the tail (:meth:`tail_values`, :meth:`tail_forces`), None for the host.
    """

    def __init__(self, grid, order: int, device=None):
        self.grid = grid
        self.order = int(order)
        self.device = device
        self.volume = float(grid.volume_element)
        self.pairs = boundary_pairs(grid)
        self._tail_key = None
        self._tail_values = None
        self._points = {}

    def points_on(self, xp, key, points):
        """``points`` as an array of ``xp``, kept per key for a device module."""

        if xp is np:
            return points
        cached = self._points.get(key)
        if cached is None:
            cached = xp.asarray(points, dtype=xp.float64)
            self._points[key] = cached
        return cached

    def tail_values(self, positions, charges) -> np.ndarray:
        """``C_L = 2 sum_a q_a/|P - R_a| - M_L[q](P)`` at the exterior points (Ry), kept for the last structure.

        The tail rows of the Poisson right-hand side are ``t = -A_IB C_L``,
        so ``u.t = -y.C_L`` whether a route adds the tail as rows or with its
        boundary values.
        """

        from .boundary import point_charge_potential

        positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        charges = np.ascontiguousarray(charges, dtype=np.float64)
        key = (positions.tobytes(), charges.tobytes())
        if key != self._tail_key:
            xp, context = _array_module(self.device)
            with context:
                exterior = self.points_on(xp, "exterior", self.pairs.exterior)
                if xp is np:
                    values = point_charge_potential(exterior, positions, charges)
                else:
                    values = coulomb_sums(exterior, positions, charges, xp)
                values = _host(values - point_charge_series(exterior, positions, charges, self.order, xp))
            self._tail_key, self._tail_values = key, values
        return self._tail_values

    def exterior_field(self, layout: FieldLayout, values) -> np.ndarray:
        """``y = A_IB^T u`` at the exterior points from a field ``u`` held in ``layout``."""

        pairs = self.pairs
        weights = pairs.coefficients * layout.at_rows(values, pairs.rows)
        return np.bincount(pairs.exterior_index, weights=weights, minlength=pairs.exterior_count)

    def moments(self, y: np.ndarray, xp=np) -> np.ndarray:
        """``P_lm = sum_b y_b Y_lm(b) / r_b^(l+1)`` for ``m >= 0``, shape ``(L+1, L+1)`` (a host array).

        ``xp`` is the array module of the sums (CuPy on the current device).
        With ``r^l Y_lm = N_lm q_l^m (x + i y)^m`` (:func:`_solid_recurrence`),
        ``Y_lm / r^(l+1) = N_lm q_l^m (x + i y)^m / r^(2l+1)``.
        """

        result = np.zeros((self.order + 1, self.order + 1), dtype=np.complex128)
        if self.pairs.exterior.shape[0] == 0:
            return result
        points = self.points_on(xp, "exterior", self.pairs.exterior)
        x, yy, z = (xp.ascontiguousarray(points[:, axis]) for axis in range(3))
        square = x * x + yy * yy + z * z
        inverse_square = 1.0 / square
        scaled = [xp.asarray(y, dtype=xp.float64) / xp.sqrt(square)]
        for _ in range(self.order):
            scaled.append(scaled[-1] * inverse_square)
        index, values = [], []
        for order_m, cosine, sine, degree, q in _solid_recurrence(x, yy, z, square, self.order, xp):
            weighted = scaled[degree] * q
            index.append((degree, order_m))
            values.append(_normalization(degree, order_m) * xp.stack((xp.dot(weighted, cosine), xp.dot(weighted, sine))))
        values = xp.stack(values)
        values = values.get() if hasattr(values, "get") else np.asarray(values)
        for (degree, order_m), (real, imaginary) in zip(index, values):
            result[degree, order_m] = complex(real, imaginary)
        return result

    def expansion(self, moments: np.ndarray, points, xp=np):
        """``2 h^3 sum_l 4 pi/(2l+1) r^l [conj(Y_l0) P_l0 + 2 Re sum_m>0 conj(Y_lm) P_lm]`` at ``points``.

        At the grid points this is ``M^T y``; at a position ``R`` it is
        ``h^3`` times the potential at the exterior points of a unit charge at
        ``R`` in the multipole series, summed with the weights ``y``.  With
        ``r^l conj(Y_lm) = N_lm q_l^m (x - i y)^m`` the term of ``m`` is
        ``C_m A_m + S_m B_m``, where ``(x + i y)^m = C_m + i S_m`` and ``A_m``,
        ``B_m`` sum ``q_l^m`` with the real and imaginary parts of the weighted
        moments.
        """

        points = xp.asarray(points, dtype=xp.float64)
        x, y, z = (xp.ascontiguousarray(points[:, axis]) for axis in range(3))
        square = x * x + y * y + z * z
        result = xp.zeros(points.shape[0], dtype=xp.float64)
        real = imaginary = None
        current = -1
        for order_m, cosine, sine, degree, q in _solid_recurrence(x, y, z, square, self.order, xp):
            if order_m != current:
                if real is not None:
                    result += previous_cosine * real + previous_sine * imaginary
                real = xp.zeros_like(result)
                imaginary = xp.zeros_like(result)
                current = order_m
            previous_cosine, previous_sine = cosine, sine
            moment = complex(moments[degree, order_m])
            weight = (1.0 if order_m == 0 else 2.0) * 4.0 * math.pi / (2 * degree + 1) * _normalization(degree, order_m)
            if moment.real:
                real += (weight * moment.real) * q
            if moment.imag:
                imaginary += (weight * moment.imag) * q
        if real is not None:
            result += previous_cosine * real + previous_sine * imaginary
        return 2.0 * self.volume * result

    def transpose(self, layout: FieldLayout, values, target: FieldLayout | None = None, xp=np):
        """``M^T A_IB^T u`` of a field ``u`` in ``layout``, at the points of ``target`` (``layout`` by default).

        A host array whatever ``xp`` the sums take.
        """

        target = layout if target is None else target
        moments = self.moments(self.exterior_field(layout, values), xp)
        key = ("points", None if target.reduction is None else id(target.reduction))
        values = self.expansion(moments, self.points_on(xp, key, target.points()), xp=xp)
        return values.get() if hasattr(values, "get") else values

    def tail_forces(self, layout: FieldLayout, values, positions, charges) -> np.ndarray:
        """``(h^3/2) d/dR_a [y.C_L(R)]`` (Ry/bohr) with ``y = A_IB^T u``: the explicit term of the atomic tail."""

        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        charges = np.asarray(charges, dtype=np.float64)
        y = self.exterior_field(layout, values)
        xp, context = _array_module(self.device)
        with context:
            moments = self.moments(y, xp)
            # The Coulomb part: d/dR sum_b y_b 2/|b - R| = sum_b y_b 2 (b - R)/|b - R|^3.
            exterior = self.points_on(xp, "exterior", self.pairs.exterior)
            forces = self.volume * _host(coulomb_sums(exterior, positions, charges, xp, weights=y))
        # The multipole part, a polynomial of the position: its gradient by
        # the fourth-order central difference, exact to rounding for these
        # degrees at the step below.
        step = 1.0e-3
        offsets = (2.0, 1.0, -1.0, -2.0)
        weights = np.array([-1.0, 8.0, -8.0, 1.0]) / (12.0 * step)
        stencil = np.concatenate(
            [positions[:, None, :] + (offset * step) * np.eye(3)[None, :, :] for offset in offsets], axis=1
        ).reshape(-1, 3)
        values_at = self.expansion(moments, stencil).reshape(positions.shape[0], len(offsets), 3)
        gradient = np.einsum("aoj,o->aj", values_at, weights)
        forces -= charges[:, None] * gradient
        return 0.5 * forces


class ZeroBoundaryHostSolver:
    """``A u = rho`` with zero boundary values by the host CG of :mod:`parsec_python.Hartree.poisson`."""

    def __init__(self, negative_laplacian, grid):
        self.operator = negative_laplacian
        self.layout = FieldLayout(grid)

    def __call__(self, rhs, start, settings):
        from .poisson import _conjugate_gradient

        controls = replace(settings, cg_criterion=1)
        initial = np.zeros_like(rhs) if start is None else start
        solution, converged, iterations, _, residual, _ = _conjugate_gradient(self.operator, rhs, initial, controls)
        return solution, bool(converged), int(iterations), float(residual)


class ZeroBoundaryBackendSolver:
    """``A u = rho`` by a prepared CG backend (``backend.solve``) of the full grid or of the normalized wedge of ``reduction``."""

    def __init__(self, backend, grid, reduction=None):
        self.backend = backend
        self.layout = FieldLayout(grid, reduction)
        self.roots = None if reduction is None else np.sqrt(np.asarray(reduction.multiplicities, dtype=np.float64))

    def __call__(self, rhs, start, settings):
        normalized = rhs if self.roots is None else self.roots * rhs
        initial = np.zeros_like(normalized) if start is None else (start if self.roots is None else self.roots * start)
        payload = self.backend.solve(
            np.ascontiguousarray(normalized, dtype=np.float64),
            np.ascontiguousarray(initial, dtype=np.float64),
            relative_tolerance=float(settings.relative_tolerance),
            absolute_tolerance=float(settings.absolute_tolerance),
            max_iterations=int(settings.max_iterations),
        )
        solution = np.asarray(payload["solution"], dtype=np.float64)
        if self.roots is not None:
            solution = solution / self.roots
        return (
            solution,
            bool(payload["converged"]),
            int(payload["iterations"]),
            float(payload["residual_norm"]),
        )


class ZeroBoundaryDeviceSolver:
    """``A u = rho`` by the device CG of ``--backend cupy`` (:mod:`parsec_python.acceleration.Hartree.cupy_poisson`) on the full grid."""

    def __init__(self, device_operator, grid):
        self.operator = device_operator
        self.layout = FieldLayout(grid)

    def __call__(self, rhs, start, settings):
        from ..acceleration.backends.cupy import require_cupy
        from ..acceleration.Hartree.cupy_poisson import _device_conjugate_gradient

        cp, _ = require_cupy()
        controls = replace(settings, cg_criterion=1)
        initial = np.zeros_like(rhs) if start is None else start
        solution, converged, iterations, _, residual, _ = _device_conjugate_gradient(
            self.operator, cp.asarray(rhs, dtype=cp.float64), cp.array(initial, dtype=cp.float64, copy=True), controls, cp
        )
        return cp.asnumpy(solution), bool(converged), int(iterations), float(residual)


class VariationalHartree:
    """The potential ``V'`` of one grid in place of the Poisson solution (see the module).

    ``solver`` solves ``A u = rho`` (:class:`ZeroBoundaryHostSolver`,
    :class:`ZeroBoundaryBackendSolver` or :class:`ZeroBoundaryDeviceSolver`).
    :meth:`potential` takes the density and the Hartree result of the route
    and returns the result with ``V'``; the ``u`` of the last call starts the
    next solve.  :meth:`tail_forces` gives the explicit force term of the
    atomic tail.  The tail enters as the positions and charges of its point
    charges (:func:`tail_charges`), passed by the caller or returned by
    ``charges()`` where the route keeps the present structure (the
    accelerated driver); ``None`` stands for a boundary without a tail.  The
    atoms of the present structure are passed or returned by ``atoms()`` in
    the same way; the estimate of ``auto`` reads them.  The stencil walk of
    the boundary is made on first use.
    """

    def __init__(
        self, grid, settings, solver, *, charges=None, atoms=None, device=None, estimator=None, threshold=None
    ):
        self.grid = grid
        self.settings = settings
        self.solver = solver
        self.charges = charges
        self.atoms = atoms
        # ``device``: the CUDA device of the solid-harmonic sums and of the
        # sums of the tail (CuPy), None for the host.
        self.device = device
        # ``estimator(density, residual, atoms)``: the self-consistency force
        # term of ``forcloc.f90`` for a potential residual on the full grid.
        # Given, the first solve of a structure decides (``active``) from the
        # first-order force change it estimates whether the run takes V' (see
        # :meth:`potential`).
        self.estimator = estimator
        self.threshold = VARIATIONAL_FORCE_THRESHOLD if threshold is None else float(threshold)
        self.active = None if estimator is not None else True
        self.estimate = None
        # (estimate, taken) of every decision; the distances of the atoms from the centre at the last one
        self.decisions = []
        self._decided_radii = None
        self._adjoint = None
        # (right-hand side, u) of the last two solves and the Poisson solution of the last, for the next start
        self.history = []
        self.last_poisson = None
        # (layout, V', V_H) of the last call that returned V' (electrostatic)
        self.record = None
        self.solves = 0
        self.seconds = 0.0
        self.solve_seconds = 0.0
        self.sum_seconds = 0.0
        self.tail_seconds = 0.0
        self.decide_seconds = 0.0
        self.iterations = 0
        self.unconverged = 0

    @property
    def adjoint(self) -> HartreeAdjoint:
        if self._adjoint is None:
            self._adjoint = HartreeAdjoint(self.grid, self.settings.multipole_order, device=self.device)
        return self._adjoint

    def solve_zero(self, density, poisson=None):
        """``u = A^-1 rho`` in the layout of the solver.

        ``poisson`` is the Poisson solution ``V_H`` of the same density in that
        layout.  Since ``V_H = 8 pi u + A^-1 b`` with a right-hand side ``b``
        of the boundary rows that changes little between two solves, the CG
        starts from ``u_before + (V_H - V_H,before) / (8 pi)``; without it,
        from the chronological extrapolation of the two solves before (that of
        :class:`parsec_python.acceleration.Hartree.native_poisson.NativePoissonSolver`).
        It stops as the solve of ``V_H`` does, at ``relative_tolerance`` of
        its initial residual plus ``absolute_tolerance``.
        """

        layout = FieldLayout.of(self.grid, density)
        rhs = layout.convert(layout.values(density), self.solver.layout)
        history = [pair for pair in self.history if pair[0].shape == rhs.shape]
        start = history[-1][1] if history else None
        if (
            poisson is not None
            and self.last_poisson is not None
            and history
            and self.last_poisson.shape == rhs.shape
        ):
            start = history[-1][1] + (np.asarray(poisson, dtype=np.float64) - self.last_poisson) / (8.0 * math.pi)
        elif len(history) == 2:
            (older_rhs, older), (newer_rhs, newer) = history
            direction = newer_rhs - older_rhs
            denominator = float(np.dot(direction, direction))
            if np.isfinite(denominator) and denominator > 0.0:
                alpha = float(np.clip(np.dot(direction, rhs - newer_rhs) / denominator, -0.5, 1.5))
                predicted = newer + alpha * (newer - older)
                if np.all(np.isfinite(predicted)):
                    start = predicted
        started = perf_counter()
        solution, converged, iterations, _ = self.solver(rhs, start, self.settings)
        self.solve_seconds += perf_counter() - started
        self.history = (history + [(np.array(rhs, copy=True), solution)])[-2:]
        self.last_poisson = None if poisson is None else np.array(poisson, dtype=np.float64, copy=True)
        self.solves += 1
        self.iterations += iterations
        if not converged:
            self.unconverged += 1
        return solution

    def _moved_outward(self, atoms) -> bool:
        """Whether an atom of ``atoms`` lies more than one grid spacing farther from the centre than at the last decision.

        A run that keeps the Poisson solution decides again then: density
        that reaches the sphere is what makes V' differ.
        """

        if atoms is None or self._decided_radii is None:
            return False
        radii = np.linalg.norm(np.array([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3), axis=1)
        if radii.shape != self._decided_radii.shape:
            return True
        return bool(np.max(radii - self._decided_radii) > float(self.grid.spacing))

    def potential(self, density, result, charges=None, atoms=None):
        """``result`` of the Poisson solve of ``density`` with ``V'`` as its potential.

        ``charges`` are ``(positions, charges)`` of the atomic tail, ``None``
        without one (then ``self.charges()`` where it is given); ``atoms``
        those of the present structure (then ``self.atoms()``).  Under
        ``auto`` the first call decides whether the run takes V'; a run that
        keeps the Poisson solution decides again at the first call of a
        structure in which an atom has moved outward by more than one grid
        spacing since that decision (:meth:`_moved_outward`).
        """

        if self.estimator is not None and self.active is False:
            if atoms is None and self.atoms is not None:
                atoms = self.atoms()
            if not self._moved_outward(atoms):
                return result
            self.active = None
        if self.active is False:
            return result
        if charges is None and self.charges is not None:
            charges = self.charges()
        if atoms is None and self.atoms is not None and self.active is None:
            atoms = self.atoms()
        potential = result.potential
        output = FieldLayout.of(self.grid, potential)
        density_layout = FieldLayout.of(self.grid, density)
        started = perf_counter()
        solver_layout = self.solver.layout
        u = self.solve_zero(density, output.convert(output.values(potential), solver_layout))
        adjoint = self.adjoint
        y = adjoint.exterior_field(solver_layout, u)
        summed = perf_counter()
        # A decision of auto, which may keep the Poisson solution, sums on the host: a run that keeps it
        # compiles no device kernels for the sums.
        transpose = self._transpose(y, output, host=self.active is None)
        self.sum_seconds += perf_counter() - summed
        values = output.values(potential)
        corrected = 0.5 * (values + 8.0 * math.pi * solver_layout.convert(u, output) - transpose)
        if charges is not None:
            # u.t = -y.C_L
            tailed = perf_counter()
            tail_dot = -float(np.dot(y, adjoint.tail_values(*charges)))
            self.tail_seconds += perf_counter() - tailed
            corrected = corrected + tail_dot / (2.0 * density_layout.total(density_layout.values(density)))
        if self.active is None:
            self.active = self._decide(density, output, corrected - values, u, charges, atoms)
            if atoms is not None:
                self._decided_radii = np.linalg.norm(
                    np.array([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3), axis=1
                )
            if not self.active:
                # the run keeps the Poisson solution: only a later decision reads this object again
                self.history = []
                self.last_poisson = None
                self.record = None
                self._adjoint = None
                self.seconds += perf_counter() - started
                return result
        self.record = (output, corrected, values)
        self.seconds += perf_counter() - started
        return replace(result, potential=output.wrap(corrected, potential))

    def electrostatic(self, potential):
        """The potential of the Poisson solve where ``potential`` is the V' of the last call, else ``potential``.

        ``potential`` is held as that call returned it (a field of a
        symmetry wedge, a plain array of its layout) or on the full grid.
        """

        if self.record is None or potential is None:
            return potential
        output, corrected, poisson = self.record
        if hasattr(potential, "values") and getattr(potential, "reduction", None) is output.reduction:
            if np.array_equal(np.asarray(potential.values), corrected):
                return output.wrap(np.array(poisson, dtype=np.float64, copy=True), potential)
            return potential
        if hasattr(potential, "get") or hasattr(potential, "values"):
            return potential
        array = np.asarray(potential)
        if array.shape == corrected.shape and np.array_equal(array, corrected):
            return np.array(poisson, dtype=np.float64, copy=True)
        full = FieldLayout(self.grid)
        if output.reduction is not None and array.shape == (self.grid.size,):
            if np.array_equal(array, output.convert(corrected, full)):
                return output.convert(poisson, full)
        return potential

    def _decide(self, density, output: FieldLayout, change, u, charges, atoms) -> bool:
        """Whether V' moves the forces by more than ``threshold`` (Ry/bohr) at this density.

        The first-order estimate: the self-consistency force term of the
        change ``V' - V_H`` (the atomic densities moved with the atoms, no
        screening; it exceeds the change it estimates on most clusters) plus
        the force of the tail.
        """

        full = FieldLayout(self.grid)
        density_layout = FieldLayout.of(self.grid, density)
        values = density_layout.convert(density_layout.values(density), full)
        residual = output.convert(np.asarray(change, dtype=np.float64), full)
        residual = residual - float(np.dot(values, residual)) / float(np.sum(values))
        decided = perf_counter()
        estimate = np.asarray(self.estimator(values, residual, atoms), dtype=np.float64)
        self.decide_seconds += perf_counter() - decided
        if charges is not None:
            tailed = perf_counter()
            estimate = estimate + self.adjoint.tail_forces(self.solver.layout, u, *charges)
            self.tail_seconds += perf_counter() - tailed
        self.estimate = float(np.abs(estimate).max(initial=0.0))
        taken = self.estimate >= self.threshold
        self.decisions.append((self.estimate, taken))
        return taken

    def _transpose(self, y, output: FieldLayout, host: bool = False) -> np.ndarray:
        """``M^T y`` at the points of ``output``, summed on the device where one is given and not ``host``."""

        adjoint = self.adjoint
        if self.device is None or host:
            return adjoint.expansion(adjoint.moments(y), output.points())
        from ..acceleration.backends.cupy import require_cupy

        cp, _ = require_cupy()
        with cp.cuda.Device(int(self.device)):
            moments = adjoint.moments(y, cp)
            key = ("points", None if output.reduction is None else id(output.reduction))
            values = adjoint.expansion(moments, adjoint.points_on(cp, key, output.points()), xp=cp)
            return cp.asnumpy(values)

    def tail_forces(self, density, positions, charges) -> np.ndarray:
        """The explicit force term of the atomic tail at ``density`` (Ry/bohr)."""

        u = self.solve_zero(density)
        tailed = perf_counter()
        forces = self.adjoint.tail_forces(self.solver.layout, u, positions, charges)
        self.tail_seconds += perf_counter() - tailed
        return forces

    def report(self) -> str:
        """Whether the run took V', why, and what it cost: the solves of ``u`` and their CG steps."""

        text = "on" if self.active else "off" if self.active is False else "not decided"
        if self.estimate is not None:
            relation = "from" if self.active else "below"
            text += (
                f" (estimated force change {self.estimate:.2e} Ry/bohr at the first density of the structure,"
                f" {relation} {self.threshold:.0e}"
            )
            if len(self.decisions) > 1:
                taken = sum(1 for _, on in self.decisions if on)
                text += f"; {len(self.decisions)} decisions, {taken} on"
            text += ")"
        text += (
            f": {self.solves} solves of A u = rho, {self.iterations} CG steps; {self.seconds:.3f} s, of which"
            f" {self.solve_seconds:.3f} s the solves, {self.sum_seconds:.3f} s the solid-harmonic sums,"
            f" {self.tail_seconds:.3f} s the sums of the tail and {self.decide_seconds:.3f} s the estimate"
        )
        if self.unconverged:
            text += f", {self.unconverged} stopped at the step budget"
        return text


class VariationalHartreeSolve:
    """A Hartree solver of the accelerated driver whose results carry ``V'``.

    ``inner`` is the solver of the route (``solve(density, initial_potential,
    **controls)``); every other attribute is that of ``inner``.
    """

    def __init__(self, inner, helper: VariationalHartree):
        self.inner = inner
        self.helper = helper

    def __call__(self, density, initial_potential=None, **kwargs):
        result = self.inner(density, initial_potential, **kwargs)
        return self.helper.potential(density, result)

    def __getattr__(self, name):
        if name in ("inner", "helper"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class ConfinedVariationalHartree:
    """``V' = (V_H + G^T rho)/2`` of a slab in place of the Poisson solution ``V_H = G rho``.

    The Hartree solve of a slab (:mod:`parsec_python.Hartree.confined`) is ``A V = 8 pi rho - A_IB M rho`` with ``M``
    the sheet sums of :mod:`parsec_python.Hartree.slab_fortran` at the missing stencil points: ``G = A^-1 (8 pi -
    A_IB M)`` is not symmetric (the continuum potential of the planes is not the inverse of the finite-difference
    operator), so ``V_H`` is not the derivative of ``E_H = (h^3/2) rho.G rho``, the SCF does not make the energy
    stationary and the Hellmann-Feynman forces miss ``(h^3/2) [rho.G drho/dR - drho/dR.G rho]`` (Ge2 slab with
    density 1.1e-4 at the edge of a half-height of 6 bohr: -1.1e-5 Ry/bohr).  ``G^T rho = 8 pi u - M^T A_IB^T u``
    with ``A u = rho`` (zero boundary values), one more conjugate-gradient solve per Hartree solve.  ``rho.V' =
    rho.V_H`` at every density, so the energy functional and its bookkeeping are unchanged; the forces then need the
    transposed operator in the term of the Gaussian charges (:meth:`transpose`).  The boundary has no term that moves
    with the atoms.  Wires keep the Poisson solution.
    """

    def __init__(self, plan, negative_laplacian, settings):
        self.plan = plan
        self.negative_laplacian = negative_laplacian
        self.settings = settings
        self.active = True
        # (u, V_H) of the last solve, the start of the next; (V', V_H) of the last potential returned
        self.last = None
        self.record = None
        self.solves = 0
        self.iterations = 0
        self.unconverged = 0
        self.seconds = 0.0

    def transpose(self, density, poisson=None) -> np.ndarray:
        """``G^T rho``; ``poisson`` is ``V_H = G rho`` of the same density where known (it starts the solve of ``u``)."""

        from .poisson import _conjugate_gradient

        started = perf_counter()
        density = np.asarray(density, dtype=np.float64)
        start = np.zeros_like(density)
        if self.last is not None and self.last[0].shape == density.shape:
            start = self.last[0]
            if poisson is not None and self.last[1] is not None:
                start = start + (np.asarray(poisson, dtype=np.float64) - self.last[1]) / (8.0 * math.pi)
        u, converged, iterations, _, _, _ = _conjugate_gradient(self.negative_laplacian, density, start, self.settings)
        stencil = self.plan.stencil
        exterior = stencil.coefficients * u[stencil.rows]
        transposed = 8.0 * math.pi * u - self.plan.boundary.boundary_transpose(exterior)
        self.last = (u, None if poisson is None else np.array(poisson, dtype=np.float64, copy=True))
        self.solves += 1
        self.iterations += int(iterations)
        if not converged:
            self.unconverged += 1
        self.seconds += perf_counter() - started
        return transposed

    def potential(self, density, result):
        """``result`` of the Poisson solve of ``density`` with ``V'`` as its potential."""

        poisson = np.asarray(result.potential, dtype=np.float64)
        corrected = 0.5 * (poisson + self.transpose(density, poisson))
        self.record = (corrected, poisson)
        return replace(result, potential=corrected)

    def electrostatic(self, potential):
        """The Poisson solution where ``potential`` is the V' of the last call, else ``potential``."""

        if self.record is None or potential is None or hasattr(potential, "get"):
            return potential
        array = np.asarray(potential)
        corrected, poisson = self.record
        if array.shape == corrected.shape and np.array_equal(array, corrected):
            return np.array(poisson, dtype=np.float64, copy=True)
        return potential

    def report(self) -> str:
        text = f"{self.solves} solves of A u = rho ({self.iterations} CG steps, {self.seconds:.3f} s)"
        if self.unconverged:
            text += f", {self.unconverged} stopped at the step budget"
        return text


def confined_variational_hartree_for(problem, plan, negative_laplacian):
    """The :class:`ConfinedVariationalHartree` of a prepared slab, or ``None`` where it does not apply.

    ``PARSEC_HARTREE_VARIATIONAL`` ``on`` or ``auto`` (an unknown value is refused here, before the SCF), a slab
    (``Boundary_Conditions: slab``; a wire keeps the Poisson solution) whose forces are computed
    (:attr:`parsec_python.models.SinglePointInput.forces_wanted`), on the sheet sums of the Fortran-derived routines.
    ``auto`` takes it there without an estimate (one more solve per Hartree solve of a slab).
    """

    setting = hartree_variational_setting()
    if setting == "off" or not bool(getattr(problem, "forces_wanted", False)):
        return None
    if getattr(problem.grid, "periodicity", 3) != 2 or not hasattr(plan.boundary, "boundary_transpose"):
        return None
    return ConfinedVariationalHartree(plan, negative_laplacian, problem.hartree)


def tail_charges(system):
    """``(positions, charges)`` of the atomic tail of a prepared cluster, ``None`` where its boundary has none."""

    plan = getattr(system, "hartree_boundary", None)
    if plan is None or not plan.atomic_tail:
        return None
    from .boundary import valence_point_charges

    return valence_point_charges(system.atoms, system.pseudopotentials, system.electron_count)


def self_consistency_estimator(grid, pseudopotentials, specifications, local_forces=None):
    """``estimator(density, residual, atoms)`` of :class:`VariationalHartree`: the self-consistency force term of a residual.

    ``local_forces`` is :func:`parsec_python.Forces.local_forces` or a
    function of its arguments (the device sums of the accelerated driver).
    """

    if local_forces is None:
        from ..Forces.local import local_forces

    def estimator(density, residual, atoms):
        return local_forces(
            grid, tuple(atoms), pseudopotentials, specifications, density, residual, np.zeros_like(residual)
        ).self_consistency

    return estimator


def variational_hartree_for(problem, grid, negative_laplacian, *, pseudopotentials=None):
    """The :class:`VariationalHartree` of a prepared cluster, or ``None`` where it does not apply.

    Reads ``PARSEC_HARTREE_VARIATIONAL`` (an unknown value is refused here,
    before the SCF) and :attr:`parsec_python.models.SinglePointInput.forces_wanted`.
    Under ``auto`` the first solve of a structure decides from its estimate
    of the force change (:meth:`VariationalHartree._decide`), which needs the
    loaded ``pseudopotentials``; the system passes its atoms with each solve.
    """

    setting = hartree_variational_setting()
    if getattr(problem, "periodic_cell", None) is not None:
        return None
    if not variational_hartree_applies(
        problem.hartree, problem.grid, forces_wanted=bool(getattr(problem, "forces_wanted", False)), setting=setting
    ):
        return None
    estimator = None
    if setting == "auto" and pseudopotentials is not None:
        estimator = self_consistency_estimator(grid, pseudopotentials, problem.pseudopotentials)
    return VariationalHartree(
        grid, problem.hartree, ZeroBoundaryHostSolver(negative_laplacian, grid), estimator=estimator
    )


def electrostatic_hartree(system, potential):
    """The Hartree potential that a result of ``system`` keeps: ``V_H`` of the Poisson solve where the SCF ran with ``V'``.

    ``potential`` is the Hartree potential of the last solve of the loop (as
    it returned it, or on the full grid); anything else is returned as it is.
    """

    helper = getattr(system, "hartree_variational", None)
    if helper is None:
        return potential
    return helper.electrostatic(potential)


def hartree_tail_forces(system, result):
    """The explicit force term of the atomic tail of ``system`` at the density of ``result``, or ``None``.

    ``None`` where the system has no :class:`VariationalHartree` or its
    boundary no atomic tail.
    """

    helper = getattr(system, "hartree_variational", None)
    if helper is None or not helper.active:
        return None
    charges = tail_charges(system)
    if charges is None:
        return None
    return helper.tail_forces(result.density, *charges)


__all__ = [
    "BoundaryPairs",
    "ConfinedVariationalHartree",
    "FieldLayout",
    "HARTREE_VARIATIONAL_SWITCH",
    "HARTREE_VARIATIONAL_VALUES",
    "HartreeAdjoint",
    "VariationalHartree",
    "VariationalHartreeSolve",
    "ZeroBoundaryBackendSolver",
    "ZeroBoundaryDeviceSolver",
    "ZeroBoundaryHostSolver",
    "boundary_pairs",
    "confined_variational_hartree_for",
    "coulomb_sums",
    "electrostatic_hartree",
    "hartree_tail_forces",
    "hartree_variational_setting",
    "point_charge_series",
    "tail_charges",
    "variational_hartree_applies",
    "variational_hartree_for",
]
