"""Forces of the LDA+U term (``J = 0``), with the projector gradients.

``forcnloc_u`` of ``forcnloc.f90`` is meant to add these forces, but its
loop skips every grid point, so PARSEC's LDA+U forces are identically zero.
This module computes them instead.  For ``J = 0`` the LDA+U energy of the
converged state is ``E = (2/nspin) sum_(s,a) (U/2) [Tr N^(a,s) - Tr (N^(a,s))^2]``
(:mod:`parsec_python.Hamiltonian.hubbard`), whose derivative with respect to
``N`` is ``(2/nspin) D``.  The orbitals are eigenvectors of the Hamiltonian
that holds ``D`` (Hellmann-Feynman), and a projector depends on its atom only
through ``r - R_a``, so

``F_a = (2/nspin) sum_s sum_n f_n sum_(m,m') D^(a,s)_(m,m')
        (<grad phi_a,m|psi_n> <phi_a,m'|psi_n> + <phi_a,m|psi_n> <grad phi_a,m'|psi_n>)``

with ``grad`` the gradient with respect to the electron coordinate and
``D`` built from the final orbitals.  The gradient of a sampled projector is
``sqrt(h^3) [R_d grad Y + R_d' Y d/|d|]`` with ``R_d'`` the slope of the
linear interpolant of :func:`parsec_python.Hamiltonian.hubbard.interpolate_hubbard_radial`:
the exact derivative of the sampled projector, so the forces are the exact
derivative of the discrete energy except where a grid point crosses the
projector radius (the projector is cut where ``R_d`` is not zero: the energy
jumps) or the distance of a point crosses a sample of the radial table (the
derivative jumps).  Over finite steps the slope of the energy then differs from
the force; the smooth projectors of ``PARSEC_LDAU_SMOOTH_PROJECTORS`` (opt-in)
remove both events.

With ``J != 0`` the potential matrix of PARSEC is not symmetric and not the
derivative of its energy (see the module of the term), so there is no force
to compute and :func:`hubbard_forces` refuses.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..Hamiltonian.hubbard import (
    HUBBARD_ANGULAR_MOMENTUM,
    HUBBARD_ORBITALS,
    HubbardProjectors,
    hubbard_support,
    support_tree,
    hubbard_terms_from_orbitals,
    interpolate_hubbard_radial,
    ldau_smooth_projectors_fraction,
    periodic_replica_translations,
)
from ..models import Atom
from ..V_ion.ionic_potential import projector_center_setting, projector_gradient_distance, projector_gradient_rule
from .nonlocal_force import real_harmonics_with_gradients


def _gradient_samples(potential, distance: np.ndarray, setting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(R_d, R_d', 1/|d|)`` of a gradient sample at ``distance`` (``PARSEC_PROJECTOR_CENTER``).

    ``auto``: all three at :func:`parsec_python.V_ion.ionic_potential.projector_gradient_distance` (3e-3 bohr next to
    the atom, ``|d|`` beyond), the rule of the Kleinman-Bylander gradients; the d tables follow ``c r^2`` from their
    first sample on to 1e-5, so this changes the gradients by round-off and keeps every route on one rule.  ``tip`` and
    ``off``: ``upot.F90``'s clamp of the distance at the first radial sample, as before.  Smooth projectors
    (``PARSEC_LDAU_SMOOTH_PROJECTORS``) take that clamp under every setting: the exact derivative of their values.
    """

    if not ldau_smooth_projectors_fraction() and projector_gradient_rule(potential, setting) == "floor":
        sampled = projector_gradient_distance(distance, potential)
        radial, slope = interpolate_hubbard_radial(potential, sampled)
        return radial, slope, 1.0 / sampled
    radial, slope = interpolate_hubbard_radial(potential, distance)
    return radial, slope, 1.0 / np.maximum(distance, potential.radii[0])


def _double_grid_hubbard_gradients(
    grid, potential, position, radius: float, translations: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and ``sum_f w_f grad[R_d Y_2m](s_f)`` (``(entries, 5, 3)``, no ``sqrt(h^3)``).

    The entries are those of the double-grid projectors of
    :func:`parsec_python.Hamiltonian.hubbard.build_hubbard_projectors`; the
    gradient of a fine point is that of the ``N = 1`` builder below.
    """

    from ..V_ion.double_grid import double_grid_support

    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    spacing = float(grid.spacing)
    position = np.asarray(position, dtype=np.float64)
    setting = projector_center_setting()
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for translation in translations:
        support = double_grid_support(
            coordinates, position - translation, radius, order, spacing
        )
        if support.rows.size == 0:
            continue
        points, fine = support.radius.shape
        distance = support.radius.reshape(-1)
        relative = support.relative.reshape(-1, 3)
        radial, slope, inverse = _gradient_samples(potential, distance, setting)
        unit = relative * inverse[:, None]
        harmonics, harmonic_gradients = real_harmonics_with_gradients(
            HUBBARD_ANGULAR_MOMENTUM, unit, inverse
        )
        gradient = (
            radial[:, None, None] * harmonic_gradients
            + (slope[:, None] * harmonics)[:, :, None] * unit[:, None, :]
        ).reshape(points, fine, HUBBARD_ORBITALS, 3)
        values.append(np.einsum("pf,pfmc->pmc", support.weights, gradient))
        rows.append(support.rows)
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, HUBBARD_ORBITALS, 3))
    return np.concatenate(rows), np.concatenate(values)


def build_hubbard_projector_gradients(
    grid,
    projectors: HubbardProjectors,
    atoms: Sequence[Atom],
    potentials,
    lattice_vectors: np.ndarray | None = None,
) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
    """``grad phi_a,m`` for every column of ``projectors`` (x, y, z).

    The support entries and the radial interpolation are those of
    :func:`parsec_python.Hamiltonian.hubbard.build_hubbard_projectors`; next to an atom the distance of the gradient
    follows ``PARSEC_PROJECTOR_CENTER`` (:func:`_gradient_samples`).
    """

    setting = projector_center_setting()
    translations = periodic_replica_translations(lattice_vectors)
    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    sqrt_volume = np.sqrt(grid.volume_element)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list, list, list] = ([], [], [])
    order = int(getattr(getattr(grid, "settings", None), "double_grid_order", 1) or 1)
    # (the support of each atom from the points a tree finds near its images: the same entries bit for bit)
    tree = support_tree(coordinates) if order == 1 else None
    for block, index in enumerate(projectors.atom_indices):
        atom = atoms[index]
        potential = potentials[atom.symbol]
        if order > 1:
            # upot.F90 vylmd with ndouble > 1: the weighted sum of the
            # gradients at the fine points of each support entry.
            entry_rows, gradient_values = _double_grid_hubbard_gradients(
                grid,
                potential,
                atom.position,
                projectors.cutoff_radii[atom.symbol],
                translations,
                order,
            )
            for m in range(HUBBARD_ORBITALS):
                rows.append(entry_rows)
                columns.append(
                    np.full(entry_rows.size, HUBBARD_ORBITALS * block + m, dtype=np.int64)
                )
                for component in range(3):
                    values[component].append(sqrt_volume * gradient_values[:, m, component])
            continue
        entry_rows, relative = hubbard_support(
            coordinates,
            atom.position,
            projectors.cutoff_radii[atom.symbol],
            translations,
            tree=tree,
        )
        distance = np.linalg.norm(relative, axis=1)
        # ylm_real sees rr/max(|rr|, rs(2)), as upot.F90 clamps rnorm (tip, off); auto: the distance of the KB gradients
        radial, slope, inverse = _gradient_samples(potential, distance, setting)
        unit = relative * inverse[:, None]
        harmonics, harmonic_gradients = real_harmonics_with_gradients(
            HUBBARD_ANGULAR_MOMENTUM, unit, inverse
        )
        for m in range(HUBBARD_ORBITALS):
            gradient = sqrt_volume * (
                radial[:, None] * harmonic_gradients[:, m, :]
                + (slope * harmonics[:, m])[:, None] * unit
            )
            rows.append(entry_rows)
            columns.append(
                np.full(entry_rows.size, HUBBARD_ORBITALS * block + m, dtype=np.int64)
            )
            for component in range(3):
                values[component].append(gradient[:, component])
    shape = projectors.projectors.shape

    def assemble(component: int) -> sp.csc_matrix:
        matrix = sp.coo_matrix(
            (
                np.concatenate(values[component]),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=shape,
        ).tocsc()
        matrix.sum_duplicates()
        return matrix

    return assemble(0), assemble(1), assemble(2)


def contract_hubbard_forces(
    projectors: HubbardProjectors,
    gradients: Sequence[sp.csc_matrix],
    wavefunctions: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    potentials: np.ndarray,
    atom_count: int,
) -> np.ndarray:
    """The ``(atoms, 3)`` forces in Ry/bohr for given ``D`` of every channel.

    ``potentials`` has shape ``(spins, Hubbard atoms, 5, 5)``; the orbitals
    and occupations are given per channel, up first.
    """

    spins = len(wavefunctions)
    factor = 2.0 / spins
    forces = np.zeros((atom_count, 3))
    for spin in range(spins):
        weights = np.asarray(occupations[spin], dtype=np.float64)
        occupied = np.flatnonzero(weights != 0.0)
        if occupied.size == 0:
            continue
        vectors = np.asarray(wavefunctions[spin], dtype=np.float64)[:, occupied]
        dots = projectors.overlaps(vectors)
        matrix = np.asarray(potentials[spin], dtype=np.float64)
        for component in range(3):
            slopes = np.asarray(gradients[component].T @ vectors).reshape(dots.shape)
            # sum_n f_n sum_(m,m') D_(m,m') (g_m c_m' + c_m g_m')
            value = np.einsum(
                "n,amk,amn,akn->a", weights[occupied], matrix, slopes, dots
            ) + np.einsum(
                "n,amk,amn,akn->a", weights[occupied], matrix, dots, slopes
            )
            forces[list(projectors.atom_indices), component] += factor * value
    return forces


def hubbard_forces(
    system,
    wavefunctions: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
) -> np.ndarray:
    """LDA+U forces of a converged ``system`` from its orbitals, per channel."""

    projectors: HubbardProjectors | None = getattr(system, "hubbard", None)
    atom_count = len(system.atoms)
    if projectors is None:
        return np.zeros((atom_count, 3))
    if np.any(projectors.hubbard_j != 0.0):
        raise NotImplementedError(
            "LDA+U forces need ldaplusu_j = 0: PARSEC's J term (the P3/P4 "
            "tensors on the real d harmonics) is not the derivative of its "
            "energy, and PARSEC's own LDA+U forces are zero (forcnloc_u)"
        )
    cell = getattr(system.input, "periodic_cell", None)
    lattice = None if cell is None else cell.lattice_vectors
    if lattice is not None and getattr(system.grid, "is_confined_periodic", False):
        # A wire or slab: images along its periodic vectors only (upot.F90).
        lattice = np.asarray(lattice)[: int(system.grid.periodicity)]
    gradients = build_hubbard_projector_gradients(
        system.grid, projectors, system.atoms, system.pseudopotentials, lattice
    )
    terms = hubbard_terms_from_orbitals(projectors, wavefunctions, occupations)
    return contract_hubbard_forces(
        projectors, gradients, wavefunctions, occupations, terms.potentials, atom_count
    )


__all__ = [
    "build_hubbard_projector_gradients",
    "contract_hubbard_forces",
    "hubbard_forces",
]
