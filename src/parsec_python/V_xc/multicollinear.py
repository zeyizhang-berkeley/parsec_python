"""The multicollinear non-collinear gradient functional (``PARSEC_NONCOLLINEAR_GGA_FRAME=multicollinear``).

A collinear spin functional ``E_col[n, s]`` (``n_up,down = (n +- s)/2``) becomes non-collinear in the multicollinear
approach of Pu et al. (Phys. Rev. Research 5, 013036 (2023)) as the average over the directions ``u`` of the sphere
of an effective functional of the projection ``s_u = u . m``,

``E_MC[n, m] = < d/dr [ r E_col[n, r s_u] ] at r = 1 >_u``,

whose gradient form ``f + s df/ds + grad s . df/d(grad s)`` (their Eq. 86) is the derivative in ``r`` written out.
For a collinear ``m = s z`` the average is ``E_col[n, s]`` at every point (the derivative of ``x f(x s)`` integrated
over ``x = u.z`` in ``[-1, 1]``), and ``E_MC`` is a smooth function of ``m`` and ``grad m`` everywhere: there is no
``m/|m|`` and no ``grad |m|``, so its field stays continuous at the nodes of ``m``, where the field of the local frame
(:func:`parsec_python.V_xc.noncollinear.noncollinear_xc`) takes the direction of whatever ``m`` keeps there.

**The discrete functional.**  The derivative in ``r`` is replaced by a radial rule: with the sphere average
``A(r) = < E_col[n, r s_u] >_u`` and ``y = r**2``, ``d/dr [r A]`` at ``r = 1`` is ``L[A] = A(1) + 2 dA/dy (1)``,
which a rule on ``RADIAL_LEVELS`` radii ``r_j = 1 - j RADIAL_STEP`` and the origin reproduces for every ``A(y)`` of
degree up to ``RADIAL_LEVELS`` in ``y`` (``sum_j rho_j y_j**k = 2 k + 1``).  The sphere rule is a Lebedev rule
(``PARSEC_NONCOLLINEAR_MC_POINTS`` points, one direction of each antipodal pair evaluated: ``E_col`` is even in
``s``).  The functional is then the sum

``E_MC = sum_v W_v E_col[n, v . m] + rho_0 E_col[n, 0]``

over the vectors ``v = r_j u`` of the unit ball, ``W_v = rho_j w_u``: its potential and field,

``V = sum_v W_v (V_up + V_down)/2``,  ``B = sum_v W_v (V_up - V_down)/2 v``,

are its exact derivatives with the first derivatives of the collinear functional only (the derivative in ``r`` of the
continuous functional would need its second derivatives in the potential).  Every term is the discrete GGA of this
code (:func:`parsec_python.V_xc.pbe_spin.discrete_spin_gga`): the gradients of ``n + n_core`` and of the three
components of ``m`` are taken once, each vector ``v`` evaluates the pointwise spin functional on
``(n + n_core +- v.m)/2`` with the gradients ``(D(n + n_core) +- v.Dm)/2``, and the fluxes summed over the vectors
take one transpose of the gradient each.

**Full polarization.**  A gradient functional is not smooth at ``|zeta| = 1``: PBE's ``phi(zeta)`` has a term
``(1 - zeta)**(2/3)``, so the effective functional of a fully polarized point has an integrable singularity
``(1 - x)**(-1/3)`` on the sphere where ``u`` meets ``m``, which no direction rule integrates well (an H atom: 1e-3
Ry from the collinear functional with 50 to 194 points, and more where a direction of the rule lies along ``m``).
Where ``|m|`` approaches ``n + n_core`` the local frame is smooth and exact, and the functional takes it, blended in
the ratio ``q = |m| / (n + n_core)``:

``e = chi(q) e_local + (1 - chi(q)) e_MC``,  ``chi`` the quintic step from 0 at ``BLEND_LOWER`` to 1 at
``BLEND_UPPER``,

with the local frame in its pointwise form, ``n_up,down = (n + n_core +- |m|)/2`` and
``grad n_up,down = (D(n + n_core) +- m_hat . Dm)/2`` (``m_hat . Dm`` the gradient of ``|m|`` where ``m`` is not
zero), and the derivatives of ``chi`` in the potential and the field.  In the collinear limit both parts are the
collinear functional, the multicollinear part within its rule (model pairs of polarization up to 1 with 50 points:
energies within 5e-9 Ry, potentials within 1e-7 Ry and fields within 2e-7 Ry), and the field is continuous for every
``m``: the multicollinear part holds every point below ``BLEND_LOWER``, the nodes of ``m`` among them.  (With the
window at 0.5 and 0.8 the multicollinear part reaches polarizations where its rule converges slowly: potentials 8e-2
Ry off in the tail of a fully polarized pair.)  The rule breaks the invariance under a global rotation of ``m``
(exactly invariant: the rotations of the octahedral group), on model pairs and triads by up to 5e-7 Ry with 26
points, 1.4e-9 with 50, 5e-13 with 110; without spin-orbit coupling the rotation is soft, and an SCF loop turns it
slowly: 26 points stall O2 and H3 at a residual of 5e-8 to 7e-8, 50 to 302 points converge them alike.

The frozen core enters the channel densities as half of the core density each, as in every spin functional of this
code, and the returned potential is also the derivative in the core density (the core force).  A magnetization above
the valence density by round-off is scaled back to it (the derivatives there are those from inside).  The opt-in
exact gradient of the atomic densities beside the nuclei (``PARSEC_GGA_NUCLEAR_GRADIENT``) adds its correction ``c``
with the shares of :func:`parsec_python.V_xc.nuclear_gradient.spin_shares`, which are linear in ``v``:
``D(n + n_core) + c`` and ``Dm + m c / (n + n_core)``.

The pointwise functional is the ``sigma``-form spin PBE or PW91 of :mod:`parsec_python.V_xc.pbe_spin` and
:mod:`parsec_python.V_xc.pw91` (that of the collinear spin driver of a cluster; the collinear and axis frames of
:mod:`parsec_python.V_xc.noncollinear` evaluate PBE in the ``s_ab`` form of :func:`parsec_python.V_xc.pbe.pbe_spin_polarized`,
the same functional within 7e-10 Ry for O2).  On a CUDA device the pointwise sum is one kernel per evaluation
(:mod:`parsec_python.acceleration.V_xc.multicollinear_xc`), with the gradients and their transposes on the device
for a cluster or an orthorhombic cell and on the host for the other grids and the correction beside the nuclei
(:func:`system_multicollinear_xc`).  On the host alone the sum costs some 80 times the local frame (``auto`` then
keeps the local frame: :func:`parsec_python.V_xc.noncollinear.frame_decision`).
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass

import numpy as np

from .noncollinear import NoncollinearXCResult

MULTICOLLINEAR_POINTS_ENV = "PARSEC_NONCOLLINEAR_MC_POINTS"
MULTICOLLINEAR_POINTS_DEFAULT = 302
#: the radial rule of the derivative in r: levels r_j = 1 - j RADIAL_STEP (j < RADIAL_LEVELS) and the origin
RADIAL_LEVELS = 3
RADIAL_STEP = 1.0e-3
#: q = |m| / (n + n_core): the multicollinear functional below BLEND_LOWER, the local frame above BLEND_UPPER
BLEND_LOWER = 0.2
BLEND_UPPER = 0.4
#: vectors of one pass (the temporaries are a few dozen fields of NODE_BATCH x points)
NODE_BATCH = 8

# Lebedev rules (Lebedev and Laikov 1999) by their number of points: one point of each orbit of the octahedral group,
# (a, b, c) with a >= b >= c >= 0, and the weight of each point of the orbit (the weights of a rule sum to 1).  The
# orbits are those of scipy.integrate.lebedev_rule; the rules of 74 points and others with a negative weight are left
# out.
_LEBEDEV_ORBITS = {
    26: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.03214285714285714),
        (0.7071067811865476, 0.7071067811865476, 0.0, 0.03809523809523809),
        (1.0, 0.0, 0.0, 0.04761904761904762),
    ),
    38: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.03214285714285714),
        (0.8880738339771153, 0.4597008433809831, 0.0, 0.02857142857142857),
        (1.0, 0.0, 0.0, 0.009523809523809525),
    ),
    50: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.02109375),
        (0.7071067811865476, 0.7071067811865476, 0.0, 0.02257495590828924),
        (0.9045340337332909, 0.3015113445777636, 0.3015113445777636, 0.02017333553791887),
        (1.0, 0.0, 0.0, 0.0126984126984127),
    ),
    110: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.009793737512487511),
        (0.6904210483822922, 0.6904210483822922, 0.21595729184584844, 0.009942814891178101),
        (0.8287699812525923, 0.3956894730559419, 0.3956894730559419, 0.00959547133607096),
        (0.8781589106040661, 0.4783690288121502, 0.0, 0.009694996361663027),
        (0.9651240350865941, 0.1851156353447362, 0.1851156353447362, 0.00821173728319111),
        (1.0, 0.0, 0.0, 0.003828270494937161),
    ),
    194: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.005573383178848737),
        (0.6712973442695226, 0.6712973442695226, 0.3141969941825863, 0.005608704082587997),
        (0.7071067811865476, 0.7071067811865476, 0.0, 0.005716905949977103),
        (0.7774932193147671, 0.4446933178717437, 0.4446933178717437, 0.005518771467273614),
        (0.8360360154824589, 0.525118572443642, 0.159041710538353, 0.005530248916233094),
        (0.9125090968674737, 0.2892465627575439, 0.2892465627575439, 0.005158237711805383),
        (0.9383192181375916, 0.3457702197611283, 0.0, 0.005051846064614808),
        (0.9829723027072532, 0.1299335447650067, 0.1299335447650067, 0.004106777028169394),
        (1.0, 0.0, 0.0, 0.0017823404472446112),
    ),
    302: (
        (0.5773502691896257, 0.5773502691896257, 0.5773502691896257, 0.003599119285025571),
        (0.6566329410219612, 0.6566329410219612, 0.37103417838482095, 0.003604822601419882),
        (0.7011766416089545, 0.7011766416089545, 0.12923867271051442, 0.0036500458076772547),
        (0.7434520429875557, 0.4729054132581005, 0.4729054132581005, 0.003576729661743367),
        (0.8000727494073951, 0.5448677372580774, 0.2510034751770465, 0.003571540554273387),
        (0.8203264198277593, 0.5718955891878961, 0.0, 0.00360082093221646),
        (0.8676436245440834, 0.3515640345570105, 0.3515640345570105, 0.003449788424305883),
        (0.9024425295330004, 0.4127724083168531, 0.1233548532583327, 0.00339231220500617),
        (0.9494543172264431, 0.2219645236294178, 0.2219645236294178, 0.0031089531224136754),
        (0.964408914879206, 0.2644152887060663, 0.0, 0.002982344963171804),
        (0.9907056213794081, 0.09618308522614784, 0.09618308522614784, 0.002352101413689164),
        (1.0, 0.0, 0.0, 0.0008545911725128148),
    ),
}
MULTICOLLINEAR_POINTS_VALUES = tuple(sorted(_LEBEDEV_ORBITS))


def multicollinear_points_setting() -> int:
    """``PARSEC_NONCOLLINEAR_MC_POINTS``: the points of the Lebedev rule of the multicollinear functional, one of
    :data:`MULTICOLLINEAR_POINTS_VALUES` (default 302).  An unknown value is an error."""

    from ..fortran_numerics import switch_value

    raw = str(switch_value(MULTICOLLINEAR_POINTS_ENV, str(MULTICOLLINEAR_POINTS_DEFAULT))).strip()
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value not in _LEBEDEV_ORBITS:
        raise ValueError(
            f"{MULTICOLLINEAR_POINTS_ENV} must be one of {', '.join(str(v) for v in MULTICOLLINEAR_POINTS_VALUES)}, "
            f"not {raw!r}"
        )
    return value


def lebedev_rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    """``(directions (points, 3), weights (points,))`` of the Lebedev rule with ``points`` points (weights sum to 1),
    in a fixed order: the orbits as listed, each point of an orbit from the sorted signed permutations."""

    directions = []
    weights = []
    for a, b, c, weight in _LEBEDEV_ORBITS[points]:
        orbit = sorted(
            {
                tuple(sign * value for sign, value in zip(signs, permuted))
                for permuted in itertools.permutations((a, b, c))
                for signs in itertools.product((1.0, -1.0), repeat=3)
            }
        )
        # (-0.0 and 0.0 are one point)
        orbit = sorted({tuple(0.0 if value == 0.0 else value for value in point) for point in orbit})
        directions.extend(orbit)
        weights.extend([weight] * len(orbit))
    directions = np.array(directions, dtype=np.float64)
    if directions.shape[0] != points:
        raise AssertionError(f"the Lebedev rule of {points} points has {directions.shape[0]}")
    return directions, np.array(weights, dtype=np.float64)


def half_sphere_rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    """One direction of each antipodal pair of :func:`lebedev_rule` (``z > 0``, or ``z = 0`` and ``y > 0``, or
    ``y = z = 0`` and ``x > 0``) with twice its weight."""

    directions, weights = lebedev_rule(points)
    x, y, z = directions.T
    keep = (z > 0.0) | ((z == 0.0) & (y > 0.0)) | ((z == 0.0) & (y == 0.0) & (x > 0.0))
    if 2 * int(np.sum(keep)) != points:
        raise AssertionError("a Lebedev rule without its antipodes")
    return directions[keep], 2.0 * weights[keep]


def radial_rule(levels: int = RADIAL_LEVELS, step: float = RADIAL_STEP) -> tuple[np.ndarray, np.ndarray, float]:
    """``(radii, weights, origin weight)``: ``sum_j rho_j y_j**k = 2 k + 1`` for ``k <= levels`` with ``y = r**2``
    at the radii ``1 - j step`` and the origin (the derivative ``d/dr [r A(r)]`` at ``r = 1`` of every ``A`` of degree
    up to ``levels`` in ``r**2``)."""

    radii = 1.0 - step * np.arange(levels, dtype=np.float64)
    nodes = np.concatenate([radii * radii, [0.0]])
    matrix = np.vander(nodes, levels + 1, increasing=True).T
    moments = 2.0 * np.arange(levels + 1, dtype=np.float64) + 1.0
    weights = np.linalg.solve(matrix, moments)
    return radii, weights[:levels], float(weights[levels])


@dataclass(frozen=True)
class MulticollinearRule:
    """The vectors ``v = r_j u`` (rows) of the multicollinear sum, their weights ``W_v = rho_j w_u`` and the weight
    of the origin (the unpolarized term)."""

    points: int
    vectors: np.ndarray
    weights: np.ndarray
    origin_weight: float

    @property
    def evaluations(self) -> int:
        """Pointwise evaluations of the collinear functional per point of the multicollinear part."""

        return int(self.weights.size) + (1 if self.origin_weight else 0)


def multicollinear_rule(
    points: int | None = None, levels: int = RADIAL_LEVELS, step: float = RADIAL_STEP
) -> MulticollinearRule:
    """The rule of the multicollinear sum: the half sphere of :func:`half_sphere_rule` (``points`` from
    ``PARSEC_NONCOLLINEAR_MC_POINTS`` at every call where not given) times :func:`radial_rule`; one object per
    ``(points, levels, step)``."""

    points = multicollinear_points_setting() if points is None else int(points)
    return _rule(points, int(levels), float(step))


@functools.lru_cache(maxsize=None)
def _rule(points: int, levels: int, step: float) -> MulticollinearRule:
    directions, sphere = half_sphere_rule(points)
    radii, radial, origin = radial_rule(levels, step)
    vectors = np.concatenate([radius * directions for radius in radii])
    weights = np.concatenate([weight * sphere for weight in radial])
    vectors.setflags(write=False)
    weights.setflags(write=False)
    return MulticollinearRule(points, vectors, weights, origin)


def blend(ratio: np.ndarray, lower: float = BLEND_LOWER, upper: float = BLEND_UPPER):
    """``(chi, dchi/dq)`` of the quintic step ``6 t**5 - 15 t**4 + 10 t**3``, ``t = (q - lower)/(upper - lower)``
    clipped to ``[0, 1]``."""

    width = upper - lower
    t = np.clip((ratio - lower) / width, 0.0, 1.0)
    chi = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    slope = 30.0 * t * t * (1.0 - t) * (1.0 - t) / width
    return chi, slope


def _pointwise(partials, up, down, gradient_up, gradient_down, gradient_total, sigma_total):
    """``partials`` at the channels (any leading shape) and the two flux halves of a vector ``v``:
    ``(f, f_up, f_down, flux_sum, flux_difference)`` with ``flux_sum = f_uu g_up + f_dd g_down + 2 f_t g_total``
    (``d f / d g_total``) and ``flux_difference = f_uu g_up - f_dd g_down`` (``d f / d (D s)``)."""

    sigma_up = np.einsum("...ip,...ip->...p", gradient_up, gradient_up)
    sigma_down = np.einsum("...ip,...ip->...p", gradient_down, gradient_down)
    f, f_up, f_down, f_uu, f_dd, f_t = partials(
        up, down, sigma_up, sigma_down, np.broadcast_to(sigma_total, up.shape)
    )
    a = f_uu[..., None, :] * gradient_up
    b = f_dd[..., None, :] * gradient_down
    flux_sum = a + b + 2.0 * f_t[..., None, :] * gradient_total
    return f, f_up, f_down, flux_sum, a - b


def _multicollinear_part(partials, rule, total, magnetization, gradient_total, gradient_m):
    """The multicollinear sum at a set of points (Hartree): ``(e, de/dT, de/dm (3, p), de/dDT (3, p), de/dDm
    (3, 3, p))`` with ``T = n + n_core``, ``Dm[a]`` the gradient of ``m_a``."""

    count = total.size
    sigma_total = np.einsum("ip,ip->p", gradient_total, gradient_total)
    energy = np.zeros(count)
    d_total = np.zeros(count)
    d_m = np.zeros((3, count))
    d_gradient_total = np.zeros((3, count))
    d_gradient_m = np.zeros((3, 3, count))
    if rule.origin_weight:
        half = 0.5 * gradient_total
        f, f_up, f_down, flux_sum, _ = _pointwise(
            partials, 0.5 * total, 0.5 * total, half, half, gradient_total, sigma_total
        )
        w = rule.origin_weight
        energy += w * f
        d_total += w * 0.5 * (f_up + f_down)
        d_gradient_total += w * flux_sum
    for start in range(0, rule.weights.size, NODE_BATCH):
        vectors = rule.vectors[start : start + NODE_BATCH]
        weights = rule.weights[start : start + NODE_BATCH]
        s = vectors @ magnetization
        g = np.einsum("ka,aip->kip", vectors, gradient_m)
        f, f_up, f_down, flux_sum, flux_difference = _pointwise(
            partials,
            0.5 * (total + s),
            0.5 * (total - s),
            0.5 * (gradient_total + g),
            0.5 * (gradient_total - g),
            gradient_total,
            sigma_total,
        )
        energy += weights @ f
        d_total += weights @ (0.5 * (f_up + f_down))
        d_m += np.einsum("k,ka,kp->ap", weights, vectors, 0.5 * (f_up - f_down))
        d_gradient_total += np.einsum("k,kip->ip", weights, flux_sum)
        d_gradient_m += np.einsum("k,ka,kip->aip", weights, vectors, flux_difference)
    return energy, d_total, d_m, d_gradient_total, d_gradient_m


def _local_part(partials, total, magnetization, gradient_total, gradient_m):
    """The pointwise local frame at points with ``m != 0`` (Hartree), the outputs of :func:`_multicollinear_part`."""

    magnitude = np.sqrt(np.einsum("ap,ap->p", magnetization, magnetization))
    direction = magnetization / magnitude
    g = np.einsum("ap,aip->ip", direction, gradient_m)
    sigma_total = np.einsum("ip,ip->p", gradient_total, gradient_total)
    f, f_up, f_down, flux_sum, flux_difference = _pointwise(
        partials,
        0.5 * (total + magnitude),
        0.5 * (total - magnitude),
        0.5 * (gradient_total + g),
        0.5 * (gradient_total - g),
        gradient_total,
        sigma_total,
    )
    # d f / d m_hat_a = Dm[a] . flux_difference, through d m_hat / d m = (1 - m_hat m_hat^T) / |m|
    q = np.einsum("aip,ip->ap", gradient_m, flux_difference)
    q = q - direction * np.einsum("ap,ap->p", direction, q)
    d_m = 0.5 * (f_up - f_down) * direction + q / magnitude
    d_gradient_m = direction[:, None, :] * flux_difference[None, :, :]
    return f, 0.5 * (f_up + f_down), d_m, flux_sum, d_gradient_m


def multicollinear_xc(
    density: np.ndarray,
    magnetization: np.ndarray,
    grid,
    partials,
    core_density: np.ndarray | None = None,
    rule: MulticollinearRule | None = None,
    lower: float | None = None,
    upper: float | None = None,
    pointwise=None,
) -> NoncollinearXCResult:
    """The multicollinear functional of the module docstring (Rydberg), its gradients and their transposes on the
    host (every grid of :func:`parsec_python.V_xc.pbe.density_gradient`, with the correction of
    ``PARSEC_GGA_NUCLEAR_GRADIENT`` in its scope).

    ``partials(n_up, n_down, sigma_uu, sigma_dd, sigma)`` is the pointwise collinear spin functional in the variables
    of :func:`parsec_python.V_xc.pbe_spin.pbe_spin_energy_partials` (Hartree; any array shape), ``lower`` and
    ``upper`` the window of the blend (default ``BLEND_LOWER`` and ``BLEND_UPPER``; ``lower >= 1`` takes the
    multicollinear part at every point).  ``pointwise(T, m, DT, Dm, rule, lower, upper)`` evaluates the pointwise part
    in place of :func:`host_pointwise_sum` (the device kernel:
    :meth:`parsec_python.acceleration.V_xc.multicollinear_xc.DeviceMulticollinearXC.pointwise`).
    """

    from .nuclear_gradient import active_correction, note_flux, recording
    from .pbe import density_gradient, gradient_transpose

    rule = multicollinear_rule() if rule is None else rule
    lower = BLEND_LOWER if lower is None else lower
    upper = BLEND_UPPER if upper is None else upper
    density = np.asarray(density, dtype=np.float64)
    magnetization = np.asarray(magnetization, dtype=np.float64)
    size = density.size
    if magnetization.shape != (3, size):
        raise ValueError("magnetization must have shape (3, n_grid)")
    total = density if core_density is None else density + np.asarray(core_density, dtype=np.float64)
    # |m| <= n: a magnetization past the valence density by round-off is scaled back to it
    positive = np.maximum(density, 0.0)
    magnitude = np.sqrt(np.einsum("ap,ap->p", magnetization, magnetization))
    clipped = magnitude > positive
    if np.any(clipped):
        scale = np.where(clipped, positive / np.where(clipped, magnitude, 1.0), 1.0)
        used = magnetization * scale
        magnitude = np.where(clipped, positive, magnitude)
    else:
        used = magnetization

    gradient_total = density_gradient(total, grid)
    gradient_m = np.stack([density_gradient(used[a], grid) for a in range(3)])
    correction = active_correction(grid)
    shares = None
    if correction is not None:
        # spin_shares: the channels take (1 +- v.m/T)/2 of c, so D T + c and Dm + m c / T
        correction = np.asarray(correction, dtype=np.float64)
        shares = np.where(total > 0.0, used / np.where(total > 0.0, total, 1.0), 0.0)
        gradient_total = gradient_total + correction
        gradient_m = gradient_m + shares[:, None, :] * correction[None, :, :]

    if pointwise is None:
        energy, d_total, d_m, d_gradient_total, d_gradient_m = host_pointwise_sum(
            partials, rule, lower, upper, total, used, gradient_total, gradient_m, magnitude
        )
    else:
        energy, d_total, d_m, d_gradient_total, d_gradient_m = pointwise(
            total, used, gradient_total, gradient_m, rule, lower, upper
        )
    if correction is not None:
        # the shares m / T of c: d/dm_a += dDm[a] . c / T, d/dT -= sum_a dDm[a] . c m_a / T^2
        flux_c = np.einsum("aip,ip->ap", d_gradient_m, correction)
        inverse = np.where(total > 0.0, 1.0 / np.where(total > 0.0, total, 1.0), 0.0)
        d_m += flux_c * inverse
        d_total -= np.einsum("ap,ap->p", flux_c, shares) * inverse
        if recording(grid):
            flux = d_gradient_total + np.einsum("ap,aip->ip", shares, d_gradient_m)
            note_flux(grid, flux, gradient_transpose(flux, grid))

    # (the derivatives of a point scaled back to |m| = n are those from inside, |m| < n: the one-sided derivative of the
    # map would add the field along m to the potential at the points round-off sets past n, a fully polarized
    # density's noise; the local frame of noncollinear_xc does the same)
    potential = 2.0 * (d_total + gradient_transpose(d_gradient_total, grid))
    field = 2.0 * np.stack([d_m[a] + gradient_transpose(d_gradient_m[a], grid) for a in range(3)])
    energy_density = 2.0 * energy
    return NoncollinearXCResult(
        potential=potential,
        field=field,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


def host_pointwise_sum(partials, rule, lower, upper, total, used, gradient_total, gradient_m, magnitude=None):
    """The pointwise part of :func:`multicollinear_xc` on the host (Hartree): ``(e, de/dT, de/dm (3, n), de/dDT
    (3, n), de/dDm (3, 3, n))`` of ``T = n + n_core``, ``m`` (already within ``|m| <= n``), ``DT`` and ``Dm``; the
    multicollinear sum, the local frame and their blend.  ``mc_spin_gga_pbe`` / ``_pw91`` of the device kernels are
    the same at every point."""

    size = total.size
    if magnitude is None:
        magnitude = np.sqrt(np.einsum("ap,ap->p", used, used))
    ratio = np.where(total > 0.0, magnitude / np.where(total > 0.0, total, 1.0), 0.0)
    chi, slope = blend(ratio, lower, upper)
    energy = np.zeros(size)
    d_total = np.zeros(size)
    d_m = np.zeros((3, size))
    d_gradient_total = np.zeros((3, size))
    d_gradient_m = np.zeros((3, 3, size))
    active = total > 0.0
    mc_points = np.flatnonzero(active & (chi < 1.0))
    local_points = np.flatnonzero(active & (chi > 0.0))
    parts = {}
    for name, points, part in (
        ("mc", mc_points, lambda p: _multicollinear_part(partials, rule, total[p], used[:, p], gradient_total[:, p],
                                                          gradient_m[:, :, p])),
        ("local", local_points, lambda p: _local_part(partials, total[p], used[:, p], gradient_total[:, p],
                                                       gradient_m[:, :, p])),
    ):
        if points.size:
            parts[name] = (points, part(points))
    for name in ("mc", "local"):
        if name not in parts:
            continue
        points, (e, dt, dm, dgt, dgm) = parts[name]
        weight = chi[points] if name == "local" else 1.0 - chi[points]
        energy[points] += weight * e
        d_total[points] += weight * dt
        d_m[:, points] += weight * dm
        d_gradient_total[:, points] += weight * dgt
        d_gradient_m[:, :, points] += weight * dgm
    # the derivative of chi(|m| / T) where both parts are evaluated
    both = np.flatnonzero(active & (chi > 0.0) & (chi < 1.0))
    if both.size:
        mc_index = np.searchsorted(parts["mc"][0], both)
        local_index = np.searchsorted(parts["local"][0], both)
        difference = parts["local"][1][0][local_index] - parts["mc"][1][0][mc_index]
        factor = slope[both] * difference / total[both]
        d_total[both] -= factor * ratio[both]
        d_m[:, both] += factor * used[:, both] / magnitude[both]
    return energy, d_total, d_m, d_gradient_total, d_gradient_m


#: the pointwise collinear functionals of the multicollinear frame (V_xc.pbe_spin, V_xc.pw91)
MULTICOLLINEAR_FUNCTIONALS = ("pbe", "pw91")


def system_multicollinear_xc(system, density: np.ndarray, magnetization: np.ndarray) -> NoncollinearXCResult:
    """:func:`multicollinear_xc` of a prepared system's functional, grid and core density, in the scope of its
    density-gradient correction (``PARSEC_GGA_NUCLEAR_GRADIENT``).  With a CUDA device
    (:func:`parsec_python.acceleration.V_xc.multicollinear_xc.device_multicollinear`: CuPy and a device, not under
    ``PARSEC_CUPY_XC=off`` or ``PARSEC_DEVICE=cpu``) the pointwise sum is its kernel, and on a cluster or orthorhombic
    grid without the correction the gradients and their transposes too; the host takes the stencils of every other
    grid (oblique cells, the grids of the Fortran-derived routines) and the correction.  The same functional to
    round-off on every route."""

    from ..acceleration.V_xc.multicollinear_xc import device_multicollinear
    from .nuclear_gradient import applied, nuclear_gradient_setting

    functional = system.input.scf.xc_functional
    partials = pointwise_partials(functional)
    core = getattr(system, "core_density", None)
    rule = multicollinear_rule()
    device = device_multicollinear(functional, system.grid)
    pointwise = None if device is None else device.pointwise
    if nuclear_gradient_setting() == "off":
        if device is not None and device.stencils:
            return device(density, magnetization, core, rule, BLEND_LOWER, BLEND_UPPER)
        return multicollinear_xc(density, magnetization, system.grid, partials, core, rule, pointwise=pointwise)
    with applied(system):
        return multicollinear_xc(density, magnetization, system.grid, partials, core, rule, pointwise=pointwise)


def pointwise_partials(functional: str):
    """The pointwise spin functional of ``functional`` (``pbe`` or ``pw91``) in the variables of
    :func:`parsec_python.V_xc.pbe_spin.pbe_spin_energy_partials`."""

    if functional == "pbe":
        from .pbe_spin import pbe_spin_energy_partials

        return pbe_spin_energy_partials
    if functional == "pw91":
        from .pw91 import pw91_spin_energy_partials

        return pw91_spin_energy_partials
    raise ValueError(f"the multicollinear frame takes {' or '.join(MULTICOLLINEAR_FUNCTIONALS)}, not {functional!r}")


__all__ = [
    "BLEND_LOWER",
    "BLEND_UPPER",
    "MULTICOLLINEAR_FUNCTIONALS",
    "MULTICOLLINEAR_POINTS_ENV",
    "MULTICOLLINEAR_POINTS_VALUES",
    "MulticollinearRule",
    "RADIAL_LEVELS",
    "RADIAL_STEP",
    "blend",
    "half_sphere_rule",
    "lebedev_rule",
    "multicollinear_points_setting",
    "multicollinear_rule",
    "multicollinear_xc",
    "pointwise_partials",
    "radial_rule",
]
