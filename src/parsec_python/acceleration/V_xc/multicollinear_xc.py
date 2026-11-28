"""The multicollinear non-collinear functional (:mod:`parsec_python.V_xc.multicollinear`) on a CUDA device.

The steps of :func:`parsec_python.V_xc.multicollinear.multicollinear_xc`: the gradients of ``n + n_core`` and of the
three components of ``m``, one kernel that sums the rule's vectors and the local frame at every point with the blend
(``mc_spin_gga_pbe`` / ``mc_spin_gga_pw91`` of ``device_xc_kernels.cu``, the pointwise functionals of the device spin
GGAs), and the transposes of the gradient of the summed fluxes.  On a cluster or orthorhombic grid (the stencils of
:mod:`.device_xc`, :func:`.device_xc.device_grid_supported`) every step runs on the device (:meth:`__call__`); on
every other grid (oblique cells, the oblique grids of the Fortran-derived routines) and under
``PARSEC_GGA_NUCLEAR_GRADIENT`` the host takes the gradients and their transposes and the kernel the pointwise sum
(:meth:`DeviceMulticollinearXC.pointwise`, the hook of ``multicollinear_xc``).  The same functional as the host to
round-off (the sums in another order).  No device under ``PARSEC_CUPY_XC=off``, ``PARSEC_DEVICE=cpu`` (``--device
cpu``), or without CuPy and a CUDA device (:func:`device_available`).
"""

from __future__ import annotations

import numpy as np

from ...V_xc.noncollinear import NoncollinearXCResult

_EVALUATORS: dict = {}


class DeviceMulticollinearXC:
    """``evaluator(density, magnetization, core_density, rule, lower, upper)`` of one grid and functional on one
    device, full-grid host arrays in and out (Rydberg), where the device stencils take the grid (``stencils``); and,
    on every grid, :meth:`pointwise`."""

    def __init__(self, functional: str, grid, device_id: int | None = None) -> None:
        from ..backends.cupy import require_cupy
        from .device_xc import _StencilGeometry, _kernels, device_grid_supported

        cp, _ = require_cupy()
        self._cp = cp
        self.functional = functional
        self.grid = grid
        self.volume_element = float(grid.volume_element)
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        self.stencils = bool(device_grid_supported(grid))
        names = (f"mc_spin_gga_{functional}",) + (("xc_gradient", "xc_adjoint") if self.stencils else ())
        with cp.cuda.Device(self.device_id):
            self.kernels = _kernels(cp, self.device_id, names)
            self.geometry = _StencilGeometry(cp, grid, None, flips=False) if self.stencils else None
        if self.geometry is not None and self.geometry.count != int(grid.size):
            raise ValueError("the device stencil does not cover the grid")
        self._rule = None
        self._nodes = None

    def _rule_arrays(self, rule):
        if rule is not self._rule:
            cp = self._cp
            vectors = np.concatenate([np.asarray(rule.vectors, dtype=np.float64), np.zeros((1, 3))])
            weights = np.concatenate([np.asarray(rule.weights, dtype=np.float64), [float(rule.origin_weight)]])
            with cp.cuda.Device(self.device_id):
                self._nodes = (cp.asarray(np.ascontiguousarray(vectors)), cp.asarray(weights), int(weights.size))
            self._rule = rule
        return self._nodes

    def _sum(self, total, m, gradient_total, gradient_m, rule, lower: float, upper: float):
        """The kernel on device arrays: ``(e, de/dT, de/dm, de/dDT, de/dDm)`` (Hartree, device arrays)."""

        from .device_xc import _launch

        cp = self._cp
        count = int(total.size)
        vectors, weights, nodes = self._rule_arrays(rule)
        energy = cp.empty(count, dtype=np.float64)
        d_total = cp.empty(count, dtype=np.float64)
        d_m = cp.empty((3, count), dtype=np.float64)
        d_gradient_total = cp.empty((3, count), dtype=np.float64)
        d_gradient_m = cp.empty((3, 3, count), dtype=np.float64)
        _launch(
            self.kernels[f"mc_spin_gga_{self.functional}"],
            count,
            total,
            m,
            gradient_total,
            gradient_m,
            np.int32(nodes),
            vectors,
            weights,
            np.float64(lower),
            np.float64(upper),
            energy,
            d_total,
            d_m,
            d_gradient_total,
            d_gradient_m,
        )
        return energy, d_total, d_m, d_gradient_total, d_gradient_m

    def pointwise(self, total, m, gradient_total, gradient_m, rule, lower: float, upper: float):
        """The pointwise sum of host arrays (``T``, ``m`` within ``|m| <= n``, ``DT`` (3, n), ``Dm`` (3, 3, n)) on the
        device: the outputs of :func:`parsec_python.V_xc.multicollinear.host_pointwise_sum` as host arrays."""

        cp = self._cp
        with cp.cuda.Device(self.device_id):
            found = self._sum(
                cp.asarray(np.ascontiguousarray(total, dtype=np.float64)),
                cp.asarray(np.ascontiguousarray(m, dtype=np.float64)),
                cp.asarray(np.ascontiguousarray(gradient_total, dtype=np.float64)),
                cp.asarray(np.ascontiguousarray(gradient_m, dtype=np.float64)),
                rule,
                lower,
                upper,
            )
            return tuple(cp.asnumpy(value) for value in found)

    def __call__(self, density, magnetization, core_density, rule, lower: float, upper: float) -> NoncollinearXCResult:
        if self.geometry is None:
            raise ValueError("the device stencils do not take this grid: multicollinear_xc with pointwise")
        cp = self._cp
        kernels = self.kernels
        geometry = self.geometry
        with cp.cuda.Device(self.device_id):
            n = cp.asarray(np.asarray(density, dtype=np.float64))
            m = cp.asarray(np.ascontiguousarray(magnetization, dtype=np.float64))
            positive = cp.maximum(n, 0.0)
            magnitude = cp.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
            clipped = magnitude > positive
            # |m| <= n: scaled back where round-off puts it past n (V_xc.multicollinear.multicollinear_xc)
            scale = cp.where(clipped, positive / cp.where(clipped, magnitude, 1.0), 1.0)
            m = cp.ascontiguousarray(m * scale[None, :])
            del magnitude, clipped, scale, positive
            total = n if core_density is None else n + cp.asarray(np.asarray(core_density, dtype=np.float64))
            total = cp.ascontiguousarray(total)
            gradient_total = geometry.gradient(cp, kernels, total)
            gradient_m = cp.stack([geometry.gradient(cp, kernels, cp.ascontiguousarray(m[a])) for a in range(3)])
            energy, d_total, d_m, d_gradient_total, d_gradient_m = self._sum(
                total, m, gradient_total, gradient_m, rule, lower, upper
            )
            del gradient_total, gradient_m, total, m, n
            potential = geometry.adjoint(cp, kernels, d_gradient_total, d_total, 2.0)
            field = cp.stack(
                [geometry.adjoint(cp, kernels, cp.ascontiguousarray(d_gradient_m[a]), cp.ascontiguousarray(d_m[a]), 2.0)
                 for a in range(3)]
            )
            energy_density = 2.0 * energy
            total_energy = self.volume_element * float(cp.sum(energy_density))
            return NoncollinearXCResult(
                potential=cp.asnumpy(potential),
                field=cp.asnumpy(field),
                energy_density=cp.asnumpy(energy_density),
                total_energy=float(total_energy),
            )


#: the availability that the ranks of an MPI run agreed on (:func:`agree_across_ranks`), ``None`` outside one
_AGREED: bool | None = None


def _process_device_available() -> bool:
    from ..resources import device_policy
    from .device_xc import device_xc_mode

    if device_xc_mode() == "off" or device_policy() == "cpu":
        return False
    try:
        from ..backends.cupy import require_cupy

        require_cupy()
    except Exception:
        return False
    return True


def device_available() -> bool:
    """Whether the multicollinear sum runs on a CUDA device: CuPy with a device, not under ``PARSEC_CUPY_XC=off`` or
    ``PARSEC_DEVICE=cpu`` (the ``--device cpu`` of the command lines); in an MPI run on every rank
    (:func:`agree_across_ranks`), so that the ranks take one frame and one route."""

    if _AGREED is not None:
        return _AGREED
    return _process_device_available()


def agree_across_ranks(comm) -> bool:
    """The logical AND of :func:`device_available` over the ranks of ``comm`` (an mpi4py communicator), kept for the
    process: called by every rank at the preparation of an MPI run, before any frame is decided.  A communicator of
    one rank forgets an agreement of before."""

    global _AGREED
    if comm is None or int(comm.Get_size()) == 1:
        _AGREED = None
        return _process_device_available()
    _AGREED = all(bool(value) for value in comm.allgather(bool(_process_device_available())))
    return _AGREED


def device_multicollinear(functional: str, grid):
    """The device evaluator of ``functional`` on ``grid`` (one per grid and device, kept), or ``None`` without a
    device (:func:`device_available`; ``PARSEC_CUPY_XC=on`` without one is an error)."""

    from .device_xc import device_xc_mode

    if not device_available():
        if device_xc_mode() == "on":
            from ..backends.cupy import require_cupy

            require_cupy()
        return None
    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    device = int(cp.cuda.Device().id)
    key = (id(grid), functional, device)
    found = _EVALUATORS.get(key)
    if found is not None and found.grid is grid:
        return found
    if len(_EVALUATORS) >= 4:
        _EVALUATORS.pop(next(iter(_EVALUATORS)))
    found = DeviceMulticollinearXC(functional, grid, device)
    _EVALUATORS[key] = found
    return found


__all__ = ["DeviceMulticollinearXC", "agree_across_ranks", "device_available", "device_multicollinear"]
