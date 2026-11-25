"""The orbital term of a meta-GGA in the sectors of a k-point (``PARSEC_KPOINT_SECTOR_META_GGA``).

A sector of a k-point (:mod:`..SCF.kpoint_sectors`) solves ``U^T H_k U`` on the orbits of the commuting
involutions ``H_k`` of its little group, ``U`` real (characters ``+-1``).  A meta-GGA adds to the Bloch Hamiltonian
(:class:`parsec_python.Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian`)

``M = sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c)``  (and ``2 sum_c [W_c G_c + G_c^T W_c]`` in the orbital form
of the density gradient, ``W = de/dg``).

The operations ``O_g`` of ``H_k`` commute and square to one, so they have common real eigenvectors ``q_a`` (the
Cartesian axes for reflections), ``O_g q_a = chi_a(g) q_a``, and the derivative along ``q_a``,
``D'_a = sum_c q_ac G_c``, carries a function of sector ``Gamma`` into sector ``Gamma chi_a``:
``B_{Gamma,a} = U_{Gamma chi_a}^T D'_a U_Gamma`` is a sparse real matrix between the two sectors, the one of the
Gamma sectors (:func:`.meta_gga_device.reduce_between_sectors`).  Every operation of ``H_k`` fixes ``k`` itself
(``R k = k``), so ``k'_a = q_a . k = (R q_a) . (R k)`` vanishes wherever ``chi_a(g) = -1`` for some ``g``: the Bloch
part ``i k'_a`` lives on the directions of trivial character only, where ``B`` maps the sector into itself.  With
``q`` orthonormal (``sum_a q_ac q_ad = delta_cd``)

``U_Gamma^T M U_Gamma = sum_a (B_a + i k'_a)^H diag(vtau_a) (B_a + i k'_a) + 2 [P_a^T diag(W'_a) B_a + B_a^T diag(W'_a) P_a]``

exactly, ``vtau_a`` the orbit means of ``vtau`` on the orbits of the target sector, ``W'_a = sum_c q_ac W_c`` taken
on the orbits shared by the sector and its target (``P_a`` the restriction to them; the ``k'`` parts of the first-order
term cancel for a real ``W``).  ``W'_a`` has the character ``chi_a`` (``W(R r) = R W(r)``), so its value on an orbit
is the mean of ``chi_a(g_j) W'_a(j)`` over the orbit's rows ``j = g_j p`` (the invariant part, the value itself for a
symmetric ``W``), as ``vtau`` and the potential take their orbit means.

The kinetic-energy density of sector orbitals ``x_n`` is a sum over orbits as the density is
(``tau_w = (d/2h^3) w_k sum_n f_n sum_a |((B_a + i k'_a) x_n)_w|^2 / |O_w|`` on the target orbits of each ``a``), and
the orbital density gradient along ``q_a`` on the shared orbits is
``g'_a(j) = chi_a(g_j) (2d/h^3) w_k sum_n f_n Re[conj(x_n) (B_a x_n)]_w / |O_w|``: no field of the whole grid is formed
before the sums are expanded.

The term of a Hamiltonian (:class:`SectorKineticTerm`) applies on the host (the CPU route and ``numpy-device``) or,
on CuPy, in the strided CSR kernel of :mod:`.meta_gga_device` on the real views of the complex blocks
(:class:`DeviceSectorKineticTerm`); each reports Gershgorin's bound of its eigenvalues (``spectral_margin``), the
filter's margin, as the term of the whole grid does.  Units Ry, bohr; ``tau`` in Hartree as everywhere.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import scipy.sparse as sp

from .meta_gga_device import derivative_directions, reduce_between_sectors

_SWITCH = "PARSEC_KPOINT_SECTOR_META_GGA"
# the largest |k'_a| on a direction of a non-trivial character taken as round-off (relative to max(1, |k|))
_SHIFT_TOLERANCE = 1.0e-10


def kpoint_sector_meta_gga_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_META_GGA``: ``auto`` (default) splits the k-points of a meta-GGA loop into the sectors
    of their little groups as those of every other k-point loop (:mod:`..SCF.kpoint_sectors`), with the orbital term
    and ``tau`` reduced to the sectors (this module); ``off`` keeps one task per k-point, bit for bit as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


def _matching_row(table: np.ndarray, character: np.ndarray) -> int:
    matches = np.flatnonzero(np.all(table == character[None, :], axis=1))
    if matches.size != 1:
        raise RuntimeError("no sector of the k-point holds this character")
    return int(matches[0])


class DecompositionDerivatives:
    """What the sectors of one decomposition share: the directions ``q_a`` and characters ``chi_a``, the derivatives
    ``D'_a = sum_c q_ac G_c`` of the whole grid (the matrices ``G_c`` themselves on the Cartesian axes, no copy), and
    the check of the reduced derivatives: one random field ``f`` of the whole grid and ``D'_a f``, since
    ``U_{Gamma chi_a}^T D'_a f = B_{Gamma,a} U_Gamma^T f`` for every sector (``D'_a`` carries the part of ``f`` in
    ``Gamma`` into ``Gamma chi_a`` and no other part there).  The geometry of each sector is made once
    (:meth:`sector`)."""

    def __init__(self, decomposition, gradient, *, check: bool = True) -> None:
        self.decomposition = decomposition
        directions, characters = derivative_directions(decomposition.reduction)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.characters = np.asarray(characters, dtype=np.float64)
        self.table = np.asarray(decomposition.characters, dtype=np.float64)
        # the sector whose character is chi_a: the phases of the rows for W'_a and g'_a
        self.character_sectors = [_matching_row(self.table, self.characters[axis]) for axis in range(3)]
        matrices = [matrix if sp.isspmatrix_csr(matrix) else sp.csr_matrix(matrix) for matrix in gradient]
        self.rotated: list[sp.csr_matrix] = []
        for axis in range(3):
            weights = self.directions[:, axis]
            nonzero = np.flatnonzero(weights != 0.0)
            if nonzero.size == 1 and weights[nonzero[0]] == 1.0:
                self.rotated.append(matrices[int(nonzero[0])])
                continue
            rotated = None
            for component in nonzero:
                term = float(weights[component]) * matrices[int(component)]
                rotated = term if rotated is None else rotated + term
            self.rotated.append(sp.csr_matrix(rotated))
        self.check = None
        if check:
            field = np.random.default_rng(20261123).standard_normal(int(decomposition.full_size))
            self.check = (field, [matrix @ field for matrix in self.rotated])
        self._sectors: dict[int, SectorMetaGeometry] = {}
        self._lock = threading.Lock()

    def sector(self, sector: int) -> "SectorMetaGeometry":
        with self._lock:
            found = self._sectors.get(int(sector))
            if found is None:
                found = self._sectors[int(sector)] = SectorMetaGeometry(self, int(sector))
            return found


class SectorMetaGeometry:
    """The static part of the orbital term of one sector: ``B_a``, ``B_a^T``, their targets and the shared orbits.

    Independent of ``k`` (the shifts ``k'_a`` are taken per k-point, :meth:`k_shifts`) and of the fields: the
    k-points of one decomposition share it (the cache of the plans of a structure), made by
    :meth:`DecompositionDerivatives.sector`.
    """

    def __init__(self, shared: DecompositionDerivatives, sector: int) -> None:
        decomposition = shared.decomposition
        self.sector = int(sector)
        self.directions = shared.directions
        table = shared.table
        self.matrices: list[sp.csr_matrix] = []
        self.transposes: list[sp.csr_matrix] = []
        self.targets: list[int] = []
        # the sector whose character is chi_a: the phases of the rows for W'_a and g'_a
        self.character_sectors = list(shared.character_sectors)
        self.target_orbits: list[np.ndarray] = []
        self.source_orbits = np.asarray(decomposition.sector_orbit_indices(self.sector), dtype=np.int64)
        # per direction (target rows, source rows, wedge orbits) of the orbits both sectors hold; None where the
        # target is the sector itself (every row)
        self.shared: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
        source_values = None if shared.check is None else decomposition.reduce_vector(shared.check[0], self.sector)
        for axis in range(3):
            target = _matching_row(table, table[self.sector] * shared.characters[axis])
            reduced = reduce_between_sectors(decomposition, shared.rotated[axis], self.sector, target)
            if source_values is not None:
                expected = decomposition.reduce_vector(shared.check[1][axis], target)
                error = float(np.max(np.abs(reduced @ source_values - expected), initial=0.0))
                if error > 1.0e-10 * max(1.0, float(np.max(np.abs(expected), initial=0.0))):
                    raise RuntimeError(f"the reduced derivative of sector {self.sector} is not exact ({error:.2e})")
            self.matrices.append(reduced)
            transposed = sp.csr_matrix(reduced.T)
            transposed.sort_indices()
            self.transposes.append(transposed)
            self.targets.append(int(target))
            orbits = np.asarray(decomposition.sector_orbit_indices(target), dtype=np.int64)
            self.target_orbits.append(orbits)
            if target == self.sector:
                self.shared.append(None)
            else:
                common, target_rows, source_rows = np.intersect1d(
                    orbits, self.source_orbits, assume_unique=True, return_indices=True
                )
                self.shared.append((target_rows.astype(np.int64), source_rows.astype(np.int64), common))
        self.rows = int(self.source_orbits.size)
        self._absolute: list | None = None
        self._devices: dict[Any, Any] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        """The host data of the geometry, without its lock and device copies (a worker process of the k-point sectors,
        ``SCF/kpoint_processes.py``, receives it once and makes the copies of its own device)."""

        state = dict(self.__dict__)
        state.pop("_lock", None)
        state["_devices"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def k_shifts(self, k_point) -> np.ndarray:
        """``k'_a = q_a . k``, zero on the directions whose target is another sector (checked to be round-off)."""

        k = np.asarray(k_point, dtype=np.float64).reshape(3)
        shifts = self.directions.T @ k
        limit = _SHIFT_TOLERANCE * max(1.0, float(np.max(np.abs(k))))
        for axis, target in enumerate(self.targets):
            if target != self.sector:
                if abs(float(shifts[axis])) > limit:
                    raise RuntimeError("an operation of the sectors does not fix the k-point")
                shifts[axis] = 0.0
        return shifts

    def absolute(self) -> list:
        """Per direction ``(|B_a|, |B_a|^T, |B_a| 1)`` for the bounds (made once)."""

        with self._lock:
            if self._absolute is None:
                found = []
                for matrix, transposed in zip(self.matrices, self.transposes):
                    forward = abs(matrix).tocsr()
                    found.append((forward, abs(transposed).tocsr(), np.asarray(forward.sum(axis=1)).reshape(-1)))
                self._absolute = found
            return self._absolute

    def device_parts(self, cp, device_id: int):
        """``(forward, backward, shared maps)`` of the device ``device_id``: the strided CSR of ``B_a`` and ``B_a^T``
        (:class:`.meta_gga_device.DeviceCSR`) and per direction ``(source row of each target row, target row of each
        source row)`` (int32, -1 where not shared; ``None`` where the target is the sector itself), made once."""

        from .meta_gga_device import DeviceCSR

        with self._lock:
            found = self._devices.get(int(device_id))
            if found is None:
                with cp.cuda.Device(int(device_id)):
                    forward = [DeviceCSR(cp, matrix) for matrix in self.matrices]
                    backward = [DeviceCSR(cp, matrix) for matrix in self.transposes]
                    maps = []
                    for axis, shared in enumerate(self.shared):
                        if shared is None:
                            maps.append(None)
                            continue
                        target_rows, source_rows, _ = shared
                        source_of_target = np.full(self.matrices[axis].shape[0], -1, dtype=np.int32)
                        source_of_target[target_rows] = source_rows
                        target_of_source = np.full(self.rows, -1, dtype=np.int32)
                        target_of_source[source_rows] = target_rows
                        maps.append((cp.asarray(source_of_target), cp.asarray(target_of_source)))
                found = self._devices[int(device_id)] = (forward, backward, maps)
            return found

    def device_shared_rows(self, cp, device_id: int) -> list:
        """Per direction ``(target rows, source rows)`` of the shared orbits on the device ``device_id`` (``None`` where
        the target is the sector itself), made once."""

        key = ("shared", int(device_id))
        with self._lock:
            found = self._devices.get(key)
            if found is None:
                with cp.cuda.Device(int(device_id)):
                    found = [
                        None if shared is None else (cp.asarray(shared[0]), cp.asarray(shared[1]))
                        for shared in self.shared
                    ]
                self._devices[key] = found
            return found


def sector_kinetic_fields(decomposition, directions, character_sectors, kinetic_potential, gradient_flux=None):
    """``(vtau, W')`` of one k-point and channel on the orbits of its decomposition (shared by its sectors).

    ``vtau`` the orbit means of the whole-grid field; ``W'`` (``None`` in the density form) the three components
    ``W'_a = sum_c q_ac W_c`` as orbit means of ``chi_a(g_j) W'_a(j)`` (zero on the orbits ``chi_a`` does not admit).
    """

    reduction = decomposition.reduction
    vtau = np.asarray(decomposition.invariant_wedge_values(np.asarray(kinetic_potential, dtype=np.float64)))
    if gradient_flux is None:
        return vtau, None
    multiplicities = np.asarray(reduction.multiplicities, dtype=np.float64)
    components = np.stack([np.asarray(flux, dtype=np.float64) for flux in gradient_flux])
    rotated = np.asarray(directions, dtype=np.float64).T @ components
    fluxes = []
    for axis in range(3):
        phases = np.asarray(decomposition.phases[int(character_sectors[axis])], dtype=np.float64)
        sums = np.bincount(reduction.full_to_wedge, weights=phases * rotated[axis], minlength=reduction.wedge_size)
        fluxes.append(sums / multiplicities)
    return vtau, tuple(fluxes)


class SectorKineticTerm:
    """The orbital term of one sector's Hamiltonian: :class:`SectorMetaGeometry` with the fields of the step.

    ``fields`` are the orbit values of :func:`sector_kinetic_fields`; ``k_point`` the k-point of the sector.
    """

    def __init__(self, geometry: SectorMetaGeometry, k_point, fields) -> None:
        self.geometry = geometry
        self.shifts = geometry.k_shifts(k_point)
        vtau, fluxes = fields
        self.potentials = [np.ascontiguousarray(vtau[orbits]) for orbits in geometry.target_orbits]
        self.doubled: list[np.ndarray] | None = None
        if fluxes is not None:
            doubled = []
            for axis in range(3):
                shared = geometry.shared[axis]
                orbits = geometry.source_orbits if shared is None else shared[2]
                doubled.append(np.ascontiguousarray(2.0 * fluxes[axis][orbits]))
            self.doubled = doubled
        self._margin: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.geometry.rows, self.geometry.rows)

    def apply(self, block) -> np.ndarray:
        """The term on a complex host block ``(rows, vectors)`` (a new array)."""

        x = np.ascontiguousarray(np.asarray(block, dtype=np.complex128))
        single = x.ndim == 1
        if single:
            x = x[:, None]
        geometry = self.geometry
        result = np.zeros_like(x)
        real = x.view(np.float64)
        for axis in range(3):
            matrix, transposed = geometry.matrices[axis], geometry.transposes[axis]
            potential = self.potentials[axis][:, None]
            shift = float(self.shifts[axis])
            derivative = np.ascontiguousarray(matrix @ real).view(np.complex128)
            flux = potential * derivative
            if shift != 0.0:
                flux += (1j * shift) * potential * x
            outgoing = flux
            if self.doubled is not None:
                doubled = self.doubled[axis][:, None]
                shared = geometry.shared[axis]
                outgoing = flux.copy()
                if shared is None:
                    outgoing += doubled * x
                else:
                    outgoing[shared[0]] += doubled * x[shared[1]]
            term = np.ascontiguousarray(transposed @ np.ascontiguousarray(outgoing).view(np.float64)).view(np.complex128)
            if shift != 0.0:
                term -= (1j * shift) * flux
            if self.doubled is not None:
                if shared is None:
                    term += doubled * derivative
                else:
                    term[shared[1]] += doubled * derivative[shared[0]]
            result += term
        return result[:, 0] if single else result

    @property
    def spectral_margin(self) -> float:
        """Gershgorin's bound of the term (Ry): the largest row sum of
        ``sum_a (|B_a|^T + |k'_a|) |vtau_a| (|B_a| + |k'_a|) 1`` plus those of the first-order term, which bound the row
        sums of its modulus (``|B_a + i k'_a| <= |B_a| + |k'_a|`` entrywise)."""

        if self._margin is None:
            total = np.zeros(self.geometry.rows)
            for axis, (forward, backward, row_sums) in enumerate(self.geometry.absolute()):
                shift = abs(float(self.shifts[axis]))
                weights = np.abs(self.potentials[axis]) * (row_sums + shift)
                doubled = None if self.doubled is None else np.abs(self.doubled[axis])
                shared = self.geometry.shared[axis]
                if doubled is not None:
                    if shared is None:
                        weights = weights + doubled
                    else:
                        weights = weights.copy()
                        weights[shared[0]] += doubled
                contribution = backward @ weights
                if shift:
                    contribution += shift * np.abs(self.potentials[axis]) * (row_sums + shift)
                if doubled is not None:
                    if shared is None:
                        contribution += doubled * row_sums
                    else:
                        contribution[shared[1]] += doubled * row_sums[shared[0]]
                total += contribution
            self._margin = float(total.max()) if total.size else 0.0
        return self._margin

    def on_backend(self, backend):
        """The term the device operator of ``backend`` applies: :class:`DeviceSectorKineticTerm` on CuPy, this host
        term on the NumPy double of the device code."""

        if getattr(backend, "name", None) == "cupy":
            return DeviceSectorKineticTerm(backend.cp, self, int(backend.device_id))
        return self


class DeviceSectorKineticTerm:
    """:class:`SectorKineticTerm` on one CuPy device (its static matrices uploaded once per device)."""

    def __init__(self, cp, term: SectorKineticTerm, device_id: int) -> None:
        self.cp = cp
        self.device_id = int(device_id)
        geometry = term.geometry
        self.forward, self.backward, self.maps = geometry.device_parts(cp, self.device_id)
        self.shifts = [float(value) for value in term.shifts]
        with cp.cuda.Device(self.device_id):
            self.potentials = [cp.asarray(values) for values in term.potentials]
            self.forward_weights = [None] * 3
            self.backward_weights = [None] * 3
            if term.doubled is not None:
                for axis in range(3):
                    doubled = term.doubled[axis]
                    shared = geometry.shared[axis]
                    if shared is None:
                        self.forward_weights[axis] = self.backward_weights[axis] = cp.asarray(doubled)
                        continue
                    forward = np.zeros(geometry.matrices[axis].shape[0])
                    forward[shared[0]] = doubled
                    backward = np.zeros(geometry.rows)
                    backward[shared[1]] = doubled
                    self.forward_weights[axis] = cp.asarray(forward)
                    self.backward_weights[axis] = cp.asarray(backward)
        # the bound of the host term (the same numbers on every backend)
        self.spectral_margin = float(term.spectral_margin)

    def apply(self, block):
        """The term on a complex device block ``(rows, vectors)`` (a new array of the same layout).

        Per direction one CSR pass each way on the real views (both parts at once) and, on a direction of the trivial
        character, two fused elementwise passes for its Bloch part (:func:`_bloch_kernels`): the blocks of a sector
        are small, and the passes, not their arithmetic, set the time."""

        cp = self.cp
        x = cp.ascontiguousarray(block)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        result = cp.empty_like(x)
        target = result.view(cp.float64)
        real = x.view(cp.float64)
        flux_kernel = post_kernel = None
        first = True
        for axis in range(3):
            forward, backward = self.forward[axis], self.backward[axis]
            potential = self.potentials[axis]
            shift = self.shifts[axis]
            weights = self.forward_weights[axis]
            maps = self.maps[axis]
            if weights is None and shift == 0.0:
                # vtau (B x), then B^T of it
                scaled = forward.product(real, row_scale=potential)
                backward.product(scaled, target, accumulate=not first)
                first = False
                continue
            derivative = cp.empty((forward.shape[0], real.shape[1]), dtype=cp.float64)
            add = None if weights is None else (None if maps is None else maps[0], weights, real)
            # vtau (B x) [+ 2 W' P x], keeping B x
            scaled = forward.product(real, row_scale=potential, secondary=derivative, add=add)
            if shift != 0.0:
                if flux_kernel is None:
                    flux_kernel, post_kernel = _bloch_kernels(cp)
                # the direction of the trivial character (target = the sector): vtau i k' x joins the flux
                outgoing = flux_kernel(scaled.view(cp.complex128), x, potential[:, None], np.float64(shift))
                scaled = outgoing.view(cp.float64)
            back_add = None if weights is None else (None if maps is None else maps[1], self.backward_weights[axis], derivative)
            backward.product(scaled, target, accumulate=not first, add=back_add)
            first = False
            if shift != 0.0:
                # - i k' vtau (B x + i k' x)
                post_kernel(derivative.view(cp.complex128), x, potential[:, None], np.float64(shift), result)
        return result[:, 0] if single else result


_BLOCH_KERNELS: dict = {}


def _bloch_kernels(cp):
    """The two elementwise kernels of the Bloch part of a direction of the trivial character: ``s + i k v x`` (the
    flux) and ``r += k^2 v x - i k v d`` (``- i k v (d + i k x)``, ``d = B x``), made once."""

    found = _BLOCH_KERNELS.get("kernels")
    if found is None:
        flux = cp.ElementwiseKernel(
            "complex128 s, complex128 x, float64 v, float64 k",
            "complex128 out",
            "out = s + complex<double>(0.0, k * v) * x",
            "parsec_sector_meta_flux",
        )
        post = cp.ElementwiseKernel(
            "complex128 d, complex128 x, float64 v, float64 k",
            "complex128 r",
            "r = r + complex<double>(k * k * v, 0.0) * x - complex<double>(0.0, k * v) * d",
            "parsec_sector_meta_post",
        )
        found = _BLOCH_KERNELS["kernels"] = (flux, post)
    return found


def _sum_kernels(cp):
    """``|d + i k x|^2`` (``d`` and ``x`` on the same rows), ``|d|^2`` and ``Re[conj(x) d]`` of the entries of complex
    blocks (the summands of ``tau`` and ``g'``), made once."""

    found = _BLOCH_KERNELS.get("sums")
    if found is None:
        square = cp.ElementwiseKernel(
            "complex128 d, complex128 x, float64 k",
            "float64 out",
            "complex<double> z = d + complex<double>(0.0, k) * x; out = z.real() * z.real() + z.imag() * z.imag()",
            "parsec_sector_meta_square",
        )
        modulus = cp.ElementwiseKernel(
            "complex128 d",
            "float64 out",
            "out = d.real() * d.real() + d.imag() * d.imag()",
            "parsec_sector_meta_modulus",
        )
        product = cp.ElementwiseKernel(
            "complex128 x, complex128 d",
            "float64 out",
            "out = x.real() * d.real() + x.imag() * d.imag()",
            "parsec_sector_meta_product",
        )
        found = _BLOCH_KERNELS["sums"] = (square, modulus, product)
    return found


def sector_orbital_sums(term_geometry: SectorMetaGeometry, shifts, vectors, occupations, xp=np, *, gradient=False):
    """Per direction ``sum_n f_n |((B_a + i k'_a) x_n)|^2`` on the rows of its target and (``gradient``)
    ``sum_n f_n Re[conj(x_n) (B_a x_n)]`` on the shared rows, as host arrays: ``(squares, products)``.

    ``vectors`` (sector rows by states) on the host or a CuPy device, ``occupations`` per column.
    """

    weights = np.asarray(occupations, dtype=np.float64).reshape(-1)
    selected = np.flatnonzero(weights != 0.0)
    geometry = term_geometry
    squares: list[np.ndarray] = []
    products: list[np.ndarray] = []
    if not selected.size:
        for axis in range(3):
            squares.append(np.zeros(geometry.matrices[axis].shape[0]))
            shared = geometry.shared[axis]
            products.append(np.zeros(geometry.rows if shared is None else shared[0].size))
        return squares, products
    on_device = xp is not np and not isinstance(vectors, np.ndarray)
    if on_device:
        cp = xp
        device_id = int(vectors.device.id)
        forward, _, _ = geometry.device_parts(cp, device_id)
        shared_rows = geometry.device_shared_rows(cp, device_id)
        square_kernel, modulus_kernel, product_kernel = _sum_kernels(cp)
        with cp.cuda.Device(device_id):
            block = cp.ascontiguousarray(vectors[:, cp.asarray(selected)] if selected.size < weights.size else vectors)
            block = cp.ascontiguousarray(block.astype(cp.complex128, copy=False))
            occupied = cp.asarray(weights[selected])
            real = block.view(cp.float64)
            pieces = []  # the sums of every direction, brought to the host in one copy
            for axis in range(3):
                derivative = forward[axis].product(real).view(cp.complex128)
                if gradient:
                    rows = shared_rows[axis]
                    left, right = (block, derivative) if rows is None else (block[rows[1]], derivative[rows[0]])
                    pieces.append(product_kernel(left, right) @ occupied)
                shift = float(shifts[axis])
                if shift != 0.0:
                    # a direction of the trivial character: its target is the sector itself (the rows of the block)
                    pieces.append(square_kernel(derivative, block, np.float64(shift)) @ occupied)
                else:
                    pieces.append(modulus_kernel(derivative) @ occupied)
            host = cp.asnumpy(cp.concatenate(pieces))
        start = 0
        for axis in range(3):
            if gradient:
                shared = geometry.shared[axis]
                size = geometry.rows if shared is None else int(shared[0].size)
                products.append(host[start : start + size])
                start += size
            size = int(geometry.matrices[axis].shape[0])
            squares.append(host[start : start + size])
            start += size
        return squares, products
    block = np.ascontiguousarray(np.asarray(vectors, dtype=np.complex128)[:, selected])
    occupied = weights[selected]
    real = block.view(np.float64)
    for axis in range(3):
        derivative = np.ascontiguousarray(geometry.matrices[axis] @ real).view(np.complex128)
        if gradient:
            shared = geometry.shared[axis]
            if shared is None:
                left, right = block, derivative
            else:
                left, right = block[shared[1]], derivative[shared[0]]
            products.append((left.real * right.real + left.imag * right.imag) @ occupied)
        if float(shifts[axis]) != 0.0:
            derivative = derivative + (1j * float(shifts[axis])) * block
        squares.append((derivative.real * derivative.real + derivative.imag * derivative.imag) @ occupied)
    return squares, products


__all__ = [
    "DecompositionDerivatives",
    "DeviceSectorKineticTerm",
    "SectorKineticTerm",
    "SectorMetaGeometry",
    "kpoint_sector_meta_gga_setting",
    "sector_kinetic_fields",
    "sector_orbital_sums",
]
