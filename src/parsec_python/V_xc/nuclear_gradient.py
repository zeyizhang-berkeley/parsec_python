"""The finite-difference density gradient of the gradient functionals, corrected near the nuclei.

A gradient functional on the grid sums ``E_xc = h^3 sum_i f(n_i, |g_i|^2)`` with ``g = D n``, the centred
finite-difference gradient of the active grid (:func:`parsec_python.V_xc.pbe.density_gradient`).  Where a grid point
lies on or near a nucleus whose pseudo-valence density has a deep hole there (the d shells of Cu, Ag, Au: the density
grows like ``r^4`` into a shell that the grid cannot resolve), the stencil of that point reaches the steep shell of its
neighbours, ``D n`` is far from the gradient of the density, and the reduced gradient of that point swings with the
position of the atom on a scale of ``1e-3`` bohr.  PARSEC 1.5 sums the same finite-difference gradient
(``exc.f90``, ``gradmvs``).

``PARSEC_GGA_NUCLEAR_GRADIENT=on`` replaces the finite-difference gradient of the superposed atomic densities by
their exact gradient:

``g = D n + c``,  ``c = grad M - D M``,  ``M(r) = sum_a tau(|r - R_a|) m_a(|r - R_a|)``,

with ``m_a`` the spherical pseudo-atomic density of the species (``sum_l f_l u_l^2 / 4 pi r^2`` of the table, with its
core-correction density where it has one, as the functional sees the core), a least-squares quintic spline in ``r^2``
(knots 0.1 bohr apart: smooth at the nucleus and free of the round-off of the table's first samples), and
``tau`` a C4 step from 1 at ``TAPER_INNER`` to 0 at ``TAPER_OUTER`` (2.5 and 4.5 bohr).  ``D`` is the grid's own gradient
(zero padding of a cluster, wraparound of a cell, the oblique and general periodic grids), applied to ``M`` once per
structure, so ``g`` is ``D (n - M) + grad M``: the atomic part of the density differentiated exactly, the rest by the
stencil.  ``c`` depends on the atoms and not on the density, so the potential keeps its form
``v = df/dn + D^T (2 df/dsigma g)`` (the derivative of the summed energy), and the energy depends on the positions
explicitly; its force on atom ``a`` is

``F_a = 2 h^3 sum_i [u_i . Hess m~_a(x_i) - (D^T u)_i grad m~_a(x_i)]``  (Ry/bohr),

``u = 2 df/dsigma g`` (Hartree) and ``x_i = r_i - R_a`` over the points within ``TAPER_OUTER`` of every image of the
atom, ``m~ = tau m`` (:meth:`NuclearGradient.forces`).  A spin channel takes the share ``n_s / n`` of ``c``
(:func:`spin_shares`; one half for equal channels), whose density dependence adds local terms to the potentials,
and its force reads ``u = sum_s (n_s / n) u_s``.

``auto`` takes the atoms of the species whose model density has a deep hole at the nucleus, ``m(0) < HOLE_DEPTH
max m`` (:func:`hole_depth`): the valence d and f shells without semicore states (Cu, Ag, Au, the other transition
metals, the lanthanides of the PBE library: ``m(0)/max m`` from 4e-4 to 0.043), where the stencil of a point at the
nucleus reaches the steep shell; the s and p species (from 0.09 on: Xe, I, Si 0.17, C 0.40, O 0.38, H and the species
with a core correction 1) keep the finite-difference gradient, and a structure without such an atom keeps its bits,
as do the spin-polarized and non-collinear runs (:func:`skips_run`).
``on`` takes every atom.  ``off`` (the default, and PARSEC 1.5's numerics under ``PARSEC_FORTRAN_NUMERICS=on``) keeps
the finite-difference gradient bit for bit.  An unknown value is refused when the input is read.

The correction reaches the evaluators through the prepared system: its XC methods (:func:`applied`) open a scope in
which the host gradient routines of this package (:func:`active_correction`) add it, and the device evaluators read it
from the system (:func:`system_correction`).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import numpy as np

NUCLEAR_GRADIENT_ENV = "PARSEC_GGA_NUCLEAR_GRADIENT"
NUCLEAR_GRADIENT_DEFAULT = "off"
NUCLEAR_GRADIENT_VALUES = ("off", "auto", "on")
#: ``auto``: the species whose model density at the nucleus is below this fraction of its largest value
HOLE_DEPTH = 0.06

#: the radii (bohr) between which the atomic model goes from itself to zero
TAPER_INNER = 2.5
TAPER_OUTER = 4.5
#: the spacing (bohr) of the knots of the least-squares spline of the atomic density, in r
KNOT_SPACING = 0.1
#: its degree: 5, a model with four continuous derivatives (the energy's second derivatives in the positions are
#: continuous where grid points cross the spheres of the knots)
SPLINE_DEGREE = 5
#: the spacing (bohr) of the samples of the table the spline is fitted to
RESAMPLE_SPACING = 0.005


def nuclear_gradient_setting() -> str:
    """``PARSEC_GGA_NUCLEAR_GRADIENT``: ``off`` (the default), ``auto`` or ``on`` (the module docstring).  An unknown
    value is an error."""

    from ..fortran_numerics import switch_value

    value = switch_value(NUCLEAR_GRADIENT_ENV, NUCLEAR_GRADIENT_DEFAULT)
    if value not in NUCLEAR_GRADIENT_VALUES:
        raise ValueError(
            f"{NUCLEAR_GRADIENT_ENV} must be one of {', '.join(NUCLEAR_GRADIENT_VALUES)}, not {value!r}"
        )
    return value


# ----- the atomic model -----------------------------------------------------------------------------------------


def _smooth_step(s: np.ndarray, s1: float, s2: float):
    """``tau(s)`` (1 for ``s <= s1``, 0 for ``s >= s2``, C4 between: the step of degree 9 whose first four
    derivatives vanish at both ends) and its first two derivatives in ``s``.

    C4, so that the energy has continuous second derivatives in the positions where grid points cross the radii of
    the step (the force term holds the second derivatives of the model, the stiffness its third)."""

    width = s2 - s1
    t = np.clip((s - s1) / width, 0.0, 1.0)
    u = 1.0 - t
    # 1 - tau = t^5 (126 - 420 t + 540 t^2 - 315 t^3 + 70 t^4), d/dt = 630 t^4 u^4, d2/dt2 = 2520 t^3 u^3 (1 - 2 t)
    tau = 1.0 - t ** 5 * (126.0 + t * (-420.0 + t * (540.0 + t * (-315.0 + 70.0 * t))))
    first = -630.0 * t ** 4 * u ** 4 / width
    second = -2520.0 * t ** 3 * u ** 3 * (1.0 - 2.0 * t) / (width * width)
    return tau, first, second


@dataclass(frozen=True)
class AtomicModel:
    """``m~(s) = tau(s) m(s)``, ``s = r^2``: the tapered spherical model density of one species."""

    spline: object
    inner: float = TAPER_INNER
    outer: float = TAPER_OUTER

    @property
    def support(self) -> float:
        return float(self.outer)

    def parts(self, s: np.ndarray):
        """``(m~, dm~/ds, d^2 m~/ds^2)`` at ``s = r^2 < outer^2``."""

        value = self.spline(s)
        first = self.spline(s, 1)
        second = self.spline(s, 2)
        tau, tau1, tau2 = _smooth_step(s, self.inner * self.inner, self.outer * self.outer)
        return (
            tau * value,
            tau1 * value + tau * first,
            tau2 * value + 2.0 * tau1 * first + tau * second,
        )


def _radial_density(potential) -> tuple[np.ndarray, np.ndarray]:
    """``(r, n(r))``: the valence density of the table's radial functions (or its stored density) plus its
    core-correction density, the density the gradient functional sees."""

    radii = np.asarray(potential.radii, dtype=np.float64)
    density = np.zeros_like(radii)
    wavefunctions = getattr(potential, "radial_wavefunctions", None) or {}
    occupations = getattr(potential, "channel_occupations", None) or {}
    if wavefunctions and any(occupations.get(l, 0.0) for l in wavefunctions):
        for l, u in wavefunctions.items():
            u = np.asarray(u, dtype=np.float64)
            density += occupations.get(l, 0.0) * u * u / (4.0 * np.pi * radii * radii)
    else:
        density += np.asarray(potential.valence_density, dtype=np.float64)
    if getattr(potential, "has_nonlinear_core_correction", False):
        density += np.asarray(potential.core_density, dtype=np.float64)
    return radii, density


_MODELS: dict = {}


def atomic_model(potential) -> AtomicModel:
    """The :class:`AtomicModel` of a pseudopotential (cached per table object)."""

    key = id(potential)
    found = _MODELS.get(key)
    if found is not None and found[0] is potential:
        return found[1]
    from scipy.interpolate import make_lsq_spline

    radii, density = _radial_density(potential)
    keep = (radii > 0.0) & np.isfinite(density)
    r, n = radii[keep], density[keep]
    knots_r = np.arange(0.0, min(TAPER_OUTER + 1.5 * KNOT_SPACING, float(r[-1])), KNOT_SPACING)
    s_knots = knots_r * knots_r
    degree = SPLINE_DEGREE
    knots = np.concatenate([[0.0] * degree, s_knots, [s_knots[-1]] * degree])
    # The table resampled every RESAMPLE_SPACING bohr (linear between its samples, its first value below the first
    # one): every knot interval then holds 20 samples, also where a logarithmic mesh is sparse (FHI tables: 0.1 bohr
    # apart at 4 bohr, where their own samples leave intervals empty and the fit singular).
    samples = np.linspace(0.0, knots_r[-1], int(round(knots_r[-1] / RESAMPLE_SPACING)) + 1)
    spline = make_lsq_spline(samples * samples, np.interp(samples, r, n), knots, k=degree)
    if not np.all(np.isfinite(spline.c)):
        raise ValueError(f"the atomic model density of {getattr(potential, 'symbol', 'a species')} is not finite")
    model = AtomicModel(spline=spline)
    if len(_MODELS) > 64:
        _MODELS.clear()
    _MODELS[key] = (potential, model)
    return model


def hole_depth(potential) -> float:
    """``m(0) / max m``: the depth of the model density of a species at its nucleus (1 where it peaks there)."""

    model = atomic_model(potential)
    radii = np.linspace(0.0, model.inner, 1001)
    values = model.spline(radii * radii)
    peak = float(np.max(values))
    return float(values[0] / peak) if peak > 0.0 else 1.0


def corrected_species(pseudopotentials, setting: str | None = None) -> tuple[str, ...]:
    """The species whose atoms the correction takes under ``setting`` (default: the present one)."""

    setting = nuclear_gradient_setting() if setting is None else setting
    if setting == "off":
        return ()
    if setting == "on":
        return tuple(sorted(pseudopotentials))
    return tuple(sorted(symbol for symbol, potential in pseudopotentials.items() if hole_depth(potential) < HOLE_DEPTH))


#: functionals without a density gradient the correction could enter (the LDAs, Hartree-Fock) and the potential
#: models whose gradient it does not enter (LB94, Casida-Salahub)
_NO_GRADIENT = frozenset({"ca", "pw92", "xalpha", "wigner", "hedin_lundqvist", "lb94", "casida_salahub", "hf"})


def takes_correction(scf) -> bool:
    """Whether the functional of the SCF settings ``scf`` sums a density gradient the correction enters: a GGA, the
    semilocal part of a hybrid, a meta-GGA in the density form of its gradient (not the orbital form)."""

    functional = str(getattr(scf, "xc_functional", "") or "").strip().lower()
    if functional in _NO_GRADIENT:
        return False
    from .libxc import is_libxc_functional

    if is_libxc_functional(functional) and not getattr(scf, "is_hybrid", False):
        from .libxc import libxc_semilocal

        try:
            if libxc_semilocal(functional).family == "lda":
                return False
        except Exception:  # noqa: BLE001 - a libxc that is not loaded: the run refuses the functional itself
            return True
    if getattr(scf, "is_meta_gga", False):
        from .meta_gga import meta_gga_gradient_form

        return meta_gga_gradient_form(scf)[0] == "density"
    return True


def skips_run(scf, setting: str | None = None) -> bool:
    """Whether ``auto`` leaves the run of the SCF settings ``scf`` to the finite-difference gradient (its bits): the
    spin-polarized and non-collinear runs.  At the grid point beside the nucleus of a polarized d-shell atom the
    minority density is ~1e-12 of the majority one (the d shell of both channels vanishes there, the s electron is in
    one): with half of the correction in each channel the SCF loop stalled (an Ag atom 5e-4 and 2e-3 bohr from a grid
    point, PBE: 400 iterations at a residual of 1e-5 to 1e-4 with a constant energy, and 358, against 44 and 56 under
    ``off``), and with the share ``n_s / n`` of :func:`spin_shares`, which ``on`` takes, it converges (50 and 49) but
    the majority channel takes the correction of the whole d shell there and the energy keeps a feature on the scale
    of 1e-3 bohr (the slope over 1e-3 and 2e-3 bohr 6.4e-4 and 8.7e-4 Ry/bohr from the force, its derivative)."""

    setting = nuclear_gradient_setting() if setting is None else setting
    return setting == "auto" and bool(getattr(scf, "spin_polarized", False) or getattr(scf, "noncollinear", False))


def corrected_atoms(system) -> tuple[str, ...]:
    """The species of the atoms of a prepared system whose density gradient its run corrects (sorted)."""

    setting = nuclear_gradient_setting()
    if setting == "off":
        return ()
    scf = getattr(getattr(system, "input", None), "scf", None)
    if scf is not None and (skips_run(scf, setting) or not takes_correction(scf)):
        return ()
    species = corrected_species(system.pseudopotentials, setting)
    present = {atom.symbol for atom in system.atoms}
    return tuple(symbol for symbol in species if symbol in present)


def forces_refusal(translation) -> str | None:
    """The reason a run with forces is refused before its SCF (``None`` where it runs): the forces of a spin-orbit
    loop with collinear spin are not implemented with the correction."""

    setting = nuclear_gradient_setting()
    problem = getattr(translation, "problem", translation)
    scf = getattr(problem, "scf", None)
    if setting == "off" or scf is None or not getattr(translation, "compute_forces", False):
        return None
    if skips_run(scf, setting) or not takes_correction(scf):
        return None
    if getattr(scf, "self_consistent_spin_orbit", False) and getattr(scf, "spin_polarized", False) and not getattr(
        scf, "noncollinear", False
    ):
        return (
            f"the forces of a spin-orbit run with collinear spin are not implemented under {NUCLEAR_GRADIENT_ENV}="
            f"{setting} (set it to off)"
        )
    return None


# ----- the grid points of a ball ----------------------------------------------------------------------------------


def _cluster_ball(grid, centre: np.ndarray, radius: float):
    """``(rows, relative, distances)`` of the active points of a cluster grid within ``radius`` of ``centre`` (the
    result of :func:`parsec_python.V_ion.ionic_potential.dense_support`, from the index box around the centre)."""

    h = float(grid.spacing)
    shift = np.asarray(grid.settings.shift, dtype=np.float64)
    low = np.ceil((centre - radius) / h - shift - 1e-9).astype(np.int64)
    high = np.floor((centre + radius) / h - shift + 1e-9).astype(np.int64)
    low = np.maximum(low, np.asarray(grid.index_min, dtype=np.int64))
    high = np.minimum(high, np.asarray(grid.index_max, dtype=np.int64))
    if np.any(high < low):
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty((0, 3)), np.empty(0)
    axes = [np.arange(low[k], high[k] + 1) for k in range(3)]
    box = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    local = box - np.asarray(grid.index_min, dtype=np.int64)
    rows = np.asarray(grid.lookup)[local[:, 0], local[:, 1], local[:, 2]].astype(np.int64)
    rows = rows[rows >= 0]
    rows.sort()
    relative = np.asarray(grid.coordinates)[rows] - centre
    distances = np.sqrt(np.einsum("ij,ij->i", relative, relative))
    keep = distances <= radius
    return rows[keep], relative[keep], distances[keep]


def _balls(grid, atom_position: np.ndarray, radius: float, lattice_vectors, periodic_dimensions: int):
    """Yield ``(rows, relative)`` of every image of an atom within ``radius`` (the images of
    :func:`parsec_python.V_ion.ionic_potential._image_translations`)."""

    from ..V_ion import ionic_potential as _ionic

    position = np.asarray(atom_position, dtype=np.float64)
    periodic = hasattr(grid, "cell") and hasattr(grid, "points_per_axis")
    if not periodic:
        if hasattr(grid, "lookup") and hasattr(grid, "index_min"):
            rows, relative, _ = _cluster_ball(grid, position, radius)
        else:
            rows, relative, _ = _ionic.dense_support(grid, position, radius)
        yield rows, relative
        return
    if periodic_dimensions == 2 and not getattr(grid, "is_confined_periodic", False) and lattice_vectors is not None:
        # a slab of Andrew Grieder's slab_boundary: the images along the in-plane vectors (atomic_density_images)
        translations = _ionic._periodic_image_translations(lattice_vectors, radius, 2)
    else:
        translations = _ionic._image_translations(grid, lattice_vectors, position, radius)
    for translation in translations:
        rows, relative, _ = _ionic._support_finder(grid, position + translation, radius)
        if len(rows):
            yield np.asarray(rows, dtype=np.int64), np.asarray(relative, dtype=np.float64)


# ----- the correction of one structure ----------------------------------------------------------------------------


@dataclass
class NuclearGradient:
    """The correction ``c`` (Hartree-independent: a density gradient, e/bohr^4) of one structure on one grid, and
    what its force term needs."""

    grid: object
    atoms: tuple
    models: dict
    lattice_vectors: np.ndarray | None
    periodic_dimensions: int
    field: np.ndarray
    seconds: float = 0.0
    _device: dict = field(default_factory=dict)

    def forces(self, flux: np.ndarray, divergence: np.ndarray) -> np.ndarray:
        """``(atoms, 3)`` Ry/bohr: ``-dE/dR_a`` at fixed density (the module docstring).

        ``flux`` is ``u = dF/dc`` per point (Hartree; ``2 df/dsigma g`` of an unpolarized functional, the average of
        the two channels' fluxes of a spin one), ``divergence`` is ``D^T u`` on the same points (the grid's exact
        transpose, the part ``v - df/dn`` of the Hartree potential).  Full-grid fields.
        """

        flux = np.asarray(flux, dtype=np.float64)
        divergence = np.asarray(divergence, dtype=np.float64)
        forces = np.zeros((len(self.atoms), 3), dtype=np.float64)
        for index, atom in enumerate(self.atoms):
            model = self.models.get(atom.symbol)
            if model is None:
                continue
            total = np.zeros(3, dtype=np.float64)
            for rows, relative in _balls(
                self.grid, atom.position, model.support, self.lattice_vectors, self.periodic_dimensions
            ):
                s = np.einsum("ij,ij->i", relative, relative)
                inside = s < model.support * model.support
                if not np.any(inside):
                    continue
                rows, relative, s = rows[inside], relative[inside], s[inside]
                _, first, second = model.parts(s)
                u = flux[:, rows].T  # (points, 3)
                # Hess m~ = 2 m~' I + 4 m~'' x x^T, grad m~ = 2 m~' x
                projection = np.einsum("ij,ij->i", u, relative)
                total += np.einsum("i,ij->j", 2.0 * first, u)
                total += np.einsum("i,ij->j", 4.0 * second * projection, relative)
                total -= np.einsum("i,ij->j", 2.0 * first * divergence[rows], relative)
            forces[index] = 2.0 * float(self.grid.volume_element) * total
        return forces

    def device_values(self, cp, rows=None, device_id: int = 0):
        """The correction on a device (CuPy), at ``rows`` of the full grid where given (the representative points
        of the orbits of a reducer), cached per device and row set."""

        key = (int(device_id), None if rows is None else id(rows))
        found = self._device.get(key)
        if found is not None and (rows is None or found[0] is rows):
            return found[1]
        values = self.field if rows is None else self.field[:, np.asarray(rows, dtype=np.int64)]
        array = cp.asarray(np.ascontiguousarray(values))
        self._device[key] = (rows, array)
        return array


def build_correction(
    grid, atoms, pseudopotentials, lattice_vectors=None, periodic_dimensions: int = 3, species=None
) -> NuclearGradient:
    """``c = grad M - D M`` of the atoms on the grid (the module docstring), of the atoms of ``species`` where given."""

    from time import perf_counter

    from .pbe import density_gradient

    started = perf_counter()
    size = int(grid.size)
    models = {}
    total = np.zeros(size, dtype=np.float64)
    gradient = np.zeros((3, size), dtype=np.float64)
    for atom in atoms:
        if species is not None and atom.symbol not in species:
            continue
        model = models.get(atom.symbol)
        if model is None:
            model = models[atom.symbol] = atomic_model(pseudopotentials[atom.symbol])
        for rows, relative in _balls(grid, atom.position, model.support, lattice_vectors, periodic_dimensions):
            s = np.einsum("ij,ij->i", relative, relative)
            inside = s < model.support * model.support
            if not np.any(inside):
                continue
            rows, relative, s = rows[inside], relative[inside], s[inside]
            value, first, _ = model.parts(s)
            # (the rows of one ball are distinct)
            total[rows] += value
            for axis in range(3):
                gradient[axis, rows] += 2.0 * first * relative[:, axis]
    field_values = gradient - density_gradient(total, grid)
    return NuclearGradient(
        grid=grid,
        atoms=tuple(atoms),
        models=models,
        lattice_vectors=None if lattice_vectors is None else np.asarray(lattice_vectors, dtype=np.float64),
        periodic_dimensions=int(periodic_dimensions),
        field=np.ascontiguousarray(field_values),
        seconds=perf_counter() - started,
    )


def _geometry(system):
    """``(lattice_vectors, periodic_dimensions)`` of a prepared system."""

    cell = getattr(getattr(system, "input", None), "periodic_cell", None)
    if cell is None:
        return None, 3
    return np.asarray(cell.lattice_vectors, dtype=np.float64), int(getattr(cell, "dimensions", 3) or 3)


_CACHE: list = []


def system_correction(system) -> NuclearGradient | None:
    """The correction of a prepared system's structure, ``None`` under ``off`` and where no atom takes it (``auto``
    without a species of :func:`corrected_species`); cached per grid, positions and setting."""

    setting = nuclear_gradient_setting()
    if setting == "off":
        return None
    scf = getattr(getattr(system, "input", None), "scf", None)
    if scf is not None and skips_run(scf, setting):
        return None
    grid = system.grid
    atoms = tuple(system.atoms)
    species = corrected_species(system.pseudopotentials, setting)
    if not any(atom.symbol in species for atom in atoms):
        return None
    key = (
        id(grid),
        setting,
        tuple(atom.symbol for atom in atoms),
        np.asarray([atom.position for atom in atoms], dtype=np.float64).tobytes(),
    )
    for known_key, known_grid, correction in _CACHE:
        if known_key == key and known_grid is grid:
            return correction
    lattice, dimensions = _geometry(system)
    correction = build_correction(grid, atoms, system.pseudopotentials, lattice, dimensions, species)
    _CACHE.insert(0, (key, grid, correction))
    # the present structure and the one before it (a relaxation step; their device copies go with them)
    del _CACHE[2:]
    return correction


# ----- the two spin channels ----------------------------------------------------------------------------------------


def spin_shares(up, down, xp=np):
    """``(w_up, w_down)``: the share ``n_s / n`` of the correction each spin channel takes (``1/2`` where ``n`` is not
    positive; NumPy or CuPy arrays, ``xp`` their module).

    The channels' gradients are ``D n_s + w_s c``, which add up to ``D n + c``.  Equal channels take ``1/2`` each,
    exactly.  The share follows each channel's density, so that a channel without density beside a nucleus keeps its
    finite-difference gradient there (the minority channel of a polarized d-shell atom at the grid point next to its
    nucleus, ~1e-12 of the majority density: the d shell of both channels vanishes there, the s electron is in one).
    With half of ``c`` in each channel the SCF loop of an Ag atom 5e-4 and 2e-3 bohr from a grid point stalled (400
    iterations at a residual of 1e-5 to 1e-4 with a constant energy, and 358; 44 and 56 under ``off``); with the share
    it takes 50 and 49.  The share depends on the density, so the potentials take the local terms of its derivative
    (:func:`share_potentials`).  At that point the majority channel then takes the correction of the whole d shell,
    and the energy of the polarized atom keeps a feature on the scale of 1e-3 bohr there (:func:`skips_run`).
    """

    total = up + down
    positive = total > 0.0
    safe = xp.where(positive, total, 1.0)
    w_up = xp.where(positive, xp.clip(up / safe, 0.0, 1.0), 0.5)
    return w_up, 1.0 - w_up


def share_potentials(up, down, correction, flux_up, flux_down, xp=np):
    """The local terms of the potentials (Hartree) from the density dependence of :func:`spin_shares`:
    ``d/dn_up = (u_up - u_down) . c n_down / n^2`` and ``d/dn_down = -(u_up - u_down) . c n_up / n^2``, with
    ``u_s = df/dg_s`` the flux of channel ``s`` (``(3, n)`` arrays)."""

    total = up + down
    positive = total > 0.0
    square = xp.where(positive, total * total, 1.0)
    dot = xp.sum((flux_up - flux_down) * correction, axis=0)
    return xp.where(positive, dot * down / square, 0.0), xp.where(positive, -dot * up / square, 0.0)


# ----- the scope of a system's XC evaluations ---------------------------------------------------------------------

_ACTIVE: dict = {}


class FluxRecord:
    """The flux ``u`` and ``D^T u`` (Hartree) that the gradient routines of a scope saw (for the force term)."""

    def __init__(self) -> None:
        self.flux = None
        self.divergence = None

    def add(self, flux, divergence, weight: float = 1.0) -> None:
        flux = weight * np.asarray(flux, dtype=np.float64)
        divergence = weight * np.asarray(divergence, dtype=np.float64)
        if self.flux is None:
            self.flux, self.divergence = flux, divergence
        else:
            self.flux = self.flux + flux
            self.divergence = self.divergence + divergence


class _Scope:
    def __init__(self, grid, provider, record=None) -> None:
        self.grid = grid
        self.provider = provider
        self.record = record

    def field(self):
        correction = self.provider()
        return None if correction is None else correction.field


def active_correction(grid) -> np.ndarray | None:
    """The correction ``c`` (3, size) a gradient routine on ``grid`` adds, ``None`` outside a scope of that grid."""

    if not _ACTIVE:
        return None
    scope = _ACTIVE.get(id(grid))
    if scope is None or scope.grid is not grid:
        return None
    return scope.field()


def note_flux(grid, flux, divergence, weight: float = 1.0) -> None:
    """Hand the flux of a gradient routine to the record of the scope of ``grid`` (if one records)."""

    if not _ACTIVE:
        return
    scope = _ACTIVE.get(id(grid))
    if scope is None or scope.grid is not grid or scope.record is None:
        return
    scope.record.add(flux, divergence, weight)


class _Applied:
    def __init__(self, grid, provider, record=None) -> None:
        self.grid = grid
        self.scope = _Scope(grid, provider, record)
        self.previous = None

    def __enter__(self):
        key = id(self.grid)
        self.previous = _ACTIVE.get(key)
        if self.previous is not None and self.scope.record is None:
            # a scope of this grid is open already (the evaluator of a system calling another one): keep it
            self.scope = None
            return self
        _ACTIVE[key] = self.scope
        return self

    def __exit__(self, *exc) -> None:
        if self.scope is None:
            return
        key = id(self.grid)
        if self.previous is None:
            _ACTIVE.pop(key, None)
        else:
            _ACTIVE[key] = self.previous


def applied(system, record: FluxRecord | None = None) -> _Applied:
    """The scope in which the gradient routines on ``system.grid`` add the system's correction (computed on first
    use, so an LDA never builds it)."""

    return _Applied(system.grid, lambda: system_correction(system), record)


def with_correction(method):
    """Decorate an XC method of a prepared system: it runs in :func:`applied` of the system (nothing under ``off``)."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if nuclear_gradient_setting() == "off":
            return method(self, *args, **kwargs)
        with applied(self):
            return method(self, *args, **kwargs)

    return wrapped


def recording(grid) -> bool:
    """Whether a scope of ``grid`` records the flux (the force term): the readable gradient routines then evaluate
    the functional, not a faster provider that hands no flux back."""

    if not _ACTIVE:
        return False
    scope = _ACTIVE.get(id(grid))
    return scope is not None and scope.grid is grid and scope.record is not None


# ----- the force term of a finished SCF ---------------------------------------------------------------------------


def _spin_channels(result):
    """``(up, down, form)`` of a spin result, ``None`` for another one: the channels of a :class:`SinglePointResult`
    of a spin loop (``result.spin``, evaluated with ``evaluate_spin_xc``) or the ``density_up`` and ``density_down``
    of a :class:`SpinPolarizedSinglePointResult` (the periodic spin drivers and the k-point loops, evaluated with
    ``evaluate_xc_spin_polarized``)."""

    channels = getattr(result, "spin", None)
    if channels is not None:
        up, down = (np.asarray(channel.density, dtype=np.float64) for channel in channels)
        return up, down, "spin"
    up = getattr(result, "density_up", None)
    down = getattr(result, "density_down", None)
    if up is not None and down is not None:
        return np.asarray(up, dtype=np.float64), np.asarray(down, dtype=np.float64), "spin_polarized"
    return None


def correction_forces(system, result) -> np.ndarray | None:
    """``(atoms, 3)`` Ry/bohr: the force term of the correction for the converged density of ``result``, ``None``
    under ``off`` or where no gradient routine took the correction (an LDA, the orbital form of a meta-GGA).

    The functional of the system is evaluated once more on the host at the density the local forces read, in a scope
    that records ``u = dF/dc`` and ``D^T u`` (:meth:`NuclearGradient.forces`): ``result.density`` of an unpolarized
    result, the density and magnetization of a non-collinear spinor result, the two channels of a spin result
    (:func:`_spin_channels`, with the share of :func:`spin_shares`), and the ``tau`` of a meta-GGA's record.
    ``system`` is a prepared system or an accelerated one (its ``reference``).
    """

    if nuclear_gradient_setting() == "off":
        return None
    host = getattr(system, "reference", None)
    host = system if host is None else host
    correction = system_correction(host)
    if correction is None:
        return None
    record = FluxRecord()
    meta = getattr(result, "meta_gga", None)
    if meta is not None and getattr(meta, "gradient", "density") == "orbital":
        return None
    taus = None if meta is None else tuple(np.asarray(t, dtype=np.float64) for t in meta.kinetic_energy_densities)
    channels = None if hasattr(result, "spinors") else _spin_channels(result)
    with applied(host, record):
        if hasattr(result, "spinors"):
            magnetization = getattr(result, "magnetization", None)
            if magnetization is not None:
                host.evaluate_xc_noncollinear(np.asarray(result.density), np.asarray(magnetization))
            elif getattr(host.input.scf, "spin_polarized", False):
                raise NotImplementedError(
                    "the forces of a spin-orbit run with collinear spin are not implemented under "
                    f"{NUCLEAR_GRADIENT_ENV}=on (set it to off)"
                )
            else:
                host.evaluate_xc(np.asarray(result.density))
        elif channels is not None:
            up, down, form = channels
            if taus is not None:
                host.evaluate_spin_xc(up, down, taus[0], taus[1])
            elif form == "spin_polarized":
                host.evaluate_xc_spin_polarized(up, down)
            else:
                host.evaluate_spin_xc(up, down)
        elif taus is not None:
            host.evaluate_xc(np.asarray(result.density, dtype=np.float64), tau=taus[0])
        else:
            host.evaluate_xc(np.asarray(result.density, dtype=np.float64))
    if record.flux is None:
        return None
    return correction.forces(record.flux, record.divergence)


__all__ = [
    "AtomicModel",
    "FluxRecord",
    "NUCLEAR_GRADIENT_ENV",
    "NUCLEAR_GRADIENT_VALUES",
    "NuclearGradient",
    "TAPER_INNER",
    "TAPER_OUTER",
    "active_correction",
    "applied",
    "atomic_model",
    "build_correction",
    "corrected_atoms",
    "corrected_species",
    "correction_forces",
    "forces_refusal",
    "hole_depth",
    "note_flux",
    "nuclear_gradient_setting",
    "recording",
    "share_potentials",
    "skips_run",
    "spin_shares",
    "takes_correction",
    "system_correction",
    "with_correction",
]
