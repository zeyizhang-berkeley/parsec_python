"""Forces of the self-consistent spin-orbit loop of a cluster, contracted on the device that holds its spinors.

The terms are those of :func:`parsec_python.Forces.compute_spinor_forces` (the readable path, which reads the host
spinors of the result; on the Fortran-derived routines those of
:func:`parsec_python.Forces.spin_orbit_fortran.compute_spin_orbit_forces`): the ion-ion force, the local terms of the total density with the spin-averaged potentials
(the core correction and the SCF residual), the Kleinman-Bylander projectors acting alike on both spin components
(:mod:`parsec_python.Forces.nonlocal_force`, each component at half the electrons of its state) and the spin-orbit
projectors (:func:`parsec_python.Forces.spin_orbit.spin_orbit_forces`), then the TS and Hartree-tail terms, the
average over the point group where asked for and the net force removed.  What differs is where the sums are formed:

* the local terms are the device sums of the scalar route (:func:`.local_device.cupy_local_forces`);
* the nonlocal terms are contracted on the device that solved the spinors, a block of columns at a time, with the
  projector and gradient matrices uploaded once per force call (:class:`SpinorForceContraction`); only the
  ``(projectors, 3)`` and ``(spin-orbit atoms, 3)`` sums come back.  The products add their terms in a fixed order
  (``PARSEC_ORDERED_SUMS``), so a run gives the same forces every time;
* in the double-group sectors of the loop (``PARSEC_SOC_SECTORS``, :mod:`..SCF.soc_sectors`) each solved column
  stands for a Kramers pair ``psi``, ``T psi``.  Time reversal leaves the Kleinman-Bylander and spin-orbit energies of
  a state unchanged (``T`` commutes with both operators and with their derivatives in the atom positions), so the
  force of the pair is that of ``psi`` times the electrons of both states: each sector column is unfolded to the whole
  grid on the device and contracted once with its pair weight, half the columns of the whole-grid contraction.

``PARSEC_SOC_DEVICE_FORCES`` (:func:`soc_device_forces_setting`) decides whether the accelerated command line runs the
forces, relaxations and molecular dynamics of such a cluster with these forces and its device spinor loop, or hands
the input to the reference solver as before.
"""

from __future__ import annotations

import os
from dataclasses import replace
from time import perf_counter
from typing import Sequence

import numpy as np
import scipy.sparse as sp

SWITCH = "PARSEC_SOC_DEVICE_FORCES"
_SETTINGS = ("auto", "off")
#: Device bytes the unfolded or uploaded spinor columns of one block may take.
_BLOCK_BYTES = 512 * 1024 * 1024


def soc_device_forces_setting() -> str:
    """``PARSEC_SOC_DEVICE_FORCES``: ``auto`` (the default) runs the forces, relaxation and molecular dynamics of a
    self-consistent spin-orbit cluster (``SO_from_scratch`` neither spin-polarized nor non-collinear, either routine
    set) on the accelerated path, with the device spinor loop in its double-group sectors and the device forces of this module;
    ``off`` hands such an input to the reference solver (its CPU spinor loop on the whole grid and the readable
    forces), as before.  Another value is an error before the SCF."""

    text = os.environ.get(SWITCH, "auto").strip().lower() or "auto"
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be one of {', '.join(_SETTINGS)}, not {text!r}")
    return text


def _ordered(matrix):
    from ..Hamiltonian.complex_csr_device import ordered_device_matrix

    return ordered_device_matrix(matrix)


def _device_csr(xp, matrix, dtype):
    """``matrix`` as a CSR of the array module ``xp``.  On a device its products add in a fixed order
    (``PARSEC_ORDERED_SUMS=auto``: a real one takes a complex block through its real view); under ``off`` it is CuPy's
    own matrix, complex so that it takes complex blocks.  With NumPy it is the SciPy matrix."""

    host = sp.csr_matrix(matrix, dtype=dtype)
    host.sum_duplicates()
    host.sort_indices()
    if xp is np:
        return host
    import cupyx.scipy.sparse as device_sparse  # type: ignore[import-not-found]

    from ..Hamiltonian.complex_csr_device import ordered_sums_setting

    if ordered_sums_setting() != "auto" and dtype != np.complex128:
        host = sp.csr_matrix(host, dtype=np.complex128)
    return _ordered(device_sparse.csr_matrix(host))


class SpinorForceContraction:
    """The device matrices of the nonlocal force of spinor columns and the sums of one force call.

    ``nonlocal_operator`` and ``gradients`` are the Kleinman-Bylander projectors of the structure and their gradients
    (:func:`parsec_python.Forces.build_nonlocal_projector_gradients`); ``spin_orbit`` the spin-orbit projectors and
    gradients of :func:`parsec_python.Forces.spin_orbit.build_spin_orbit_force_operator`, or ``None``.  The real
    Kleinman-Bylander matrices act on the real view of a complex block (the up and the down component in turn); the
    spin-orbit ones are complex.  :meth:`add` takes blocks ``(2 n, m)`` of whole-grid spinors on the current device
    with the electrons of each column; :meth:`forces` returns the ``(atoms, 3)`` sum in Ry/bohr.  ``xp`` is the array
    module (CuPy; NumPy runs the same sums on the host).
    """

    def __init__(self, xp, nonlocal_operator, gradients, spin_orbit) -> None:
        from parsec_python.Forces.nonlocal_force import gradient_columns_for

        self.xp = xp
        cp = xp
        projectors = nonlocal_operator.projectors
        self.projector_count = int(projectors.shape[1])
        if self.projector_count:
            columns = gradient_columns_for(nonlocal_operator.labels, gradients.labels)
            # <beta_q|u> = beta^T u: the real transposes (the projectors of a cluster are real)
            self.kb_overlap = _device_csr(cp, sp.csc_matrix(projectors).T, np.float64)
            self.kb_gradient = [
                _device_csr(cp, sp.csc_matrix(component)[:, columns].T, np.float64)
                for component in gradients.components
            ]
            self.kb_signs = np.asarray(nonlocal_operator.signs, dtype=np.float64)
            self.kb_owners = np.fromiter(
                (label[0] for label in nonlocal_operator.labels), dtype=np.int64, count=len(nonlocal_operator.labels)
            )
            self.kb_sums = cp.zeros((self.projector_count, 3), dtype=cp.float64)
        self.spin_orbit_atoms = 0 if spin_orbit is None else int(spin_orbit.atom_count)
        if self.spin_orbit_atoms:
            self.so_overlap = _device_csr(cp, spin_orbit.overlap, np.complex128)
            self.so_gradient = [_device_csr(cp, component, np.complex128) for component in spin_orbit.gradient]
            self.so_signs = cp.asarray(np.asarray(spin_orbit.signs, dtype=np.float64))
            self.so_owners = np.asarray(spin_orbit.owners, dtype=np.int64)
            self.so_sums = cp.zeros((self.spin_orbit_atoms, 3), dtype=cp.float64)

    def add(self, block, weights: np.ndarray) -> None:
        """Add the forces of the columns of ``block`` (``(2 n, m)`` complex, current device) holding ``weights``
        electrons each (``f_n``, or the electrons of a Kramers pair for a sector column)."""

        from parsec_python.Forces.spin_orbit import spin_orbit_coefficients

        cp = self.xp
        weights = np.asarray(weights, dtype=np.float64)
        count = int(block.shape[1])
        if weights.shape != (count,):
            raise ValueError("one weight per spinor column is required")
        if not count or not np.any(weights != 0.0):
            return
        block = cp.ascontiguousarray(block, dtype=cp.complex128)
        n = int(block.shape[0]) // 2
        up, down = block[:n], block[n:]
        if self.projector_count:
            # each component holds half the electrons of its state: 4 (f/2) s Re[conj(<beta|u>) <G|u>]
            halves = cp.asarray(0.5 * weights)
            for component in (up, down):
                overlaps = cp.conj(self.kb_overlap @ component)
                for axis, matrix in enumerate(self.kb_gradient):
                    self.kb_sums[:, axis] += cp.real(overlaps * (matrix @ component)) @ halves
        if self.spin_orbit_atoms:
            atoms = self.spin_orbit_atoms
            state_weights = cp.asarray(weights)
            overlaps = cp.stack((self.so_overlap @ up, self.so_overlap @ down)).reshape(2, atoms, 16, count)
            coefficients = spin_orbit_coefficients(cp, overlaps, self.so_signs)
            for axis, matrix in enumerate(self.so_gradient):
                slopes = cp.stack((matrix @ up, matrix @ down)).reshape(2, atoms, 16, count)
                coupling = cp.einsum("sajn,sajn->an", cp.conj(slopes), coefficients).real
                self.so_sums[:, axis] += 2.0 * (coupling @ state_weights)

    def parts(self, atom_count: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``(atoms, 3)`` forces of every column added: the Kleinman-Bylander term and the spin-orbit term."""

        separable = np.zeros((atom_count, 3), dtype=np.float64)
        spin_orbit = np.zeros((atom_count, 3), dtype=np.float64)
        if self.projector_count:
            per_projector = _host(self.kb_sums) * (4.0 * self.kb_signs)[:, None]
            for axis in range(3):
                separable[:, axis] = np.bincount(self.kb_owners, weights=per_projector[:, axis], minlength=atom_count)
        if self.spin_orbit_atoms:
            per_atom = _host(self.so_sums)
            for axis in range(3):
                spin_orbit[:, axis] = np.bincount(self.so_owners, weights=per_atom[:, axis], minlength=atom_count)
        return separable, spin_orbit

    def forces(self, atom_count: int) -> np.ndarray:
        """The ``(atoms, 3)`` nonlocal force of every column added (Kleinman-Bylander plus spin-orbit)."""

        separable, spin_orbit = self.parts(atom_count)
        return separable + spin_orbit


def _host(array) -> np.ndarray:
    return np.asarray(array.get() if hasattr(array, "get") else array, dtype=np.float64)


def _block_width(rows: int) -> int:
    return max(1, int(_BLOCK_BYTES // (16 * max(int(rows), 1))))


def contract_spinors(contraction: SpinorForceContraction, spinors, occupations: np.ndarray) -> str:
    """Add the forces of a spinor result to ``contraction``; returns where its columns were read.

    ``spinors`` is what the device loop kept: the sector spinors of :class:`..SCF.soc_sectors.SectorSpinors` (each
    solved column with the electrons of its Kramers pair, unfolded on its device a block at a time), a device array of
    whole-grid spinors (its leading columns, one per state of ``occupations``) or the host spinors of the result
    (uploaded a block at a time)."""

    from ..SCF.soc_sectors import SectorSpinors

    cp = contraction.xp
    occupations = np.asarray(occupations, dtype=np.float64)
    if isinstance(spinors, SectorSpinors):
        weights = spinors.pair_weights(occupations)
        rows = 2 * spinors.parts[0].grid_size
        width = _block_width(rows)
        for parts, vectors, places in zip(spinors.parts, spinors.vectors, spinors.places):
            places = np.asarray(places)
            # the columns of this sector among the lowest pairs, with the electrons of their pairs
            kept = np.flatnonzero((places >= 0) & (places < weights.size))
            kept = kept[weights[places[kept]] != 0.0]
            for start in range(0, int(kept.size), width):
                chosen = kept[start : start + width]
                if np.all(np.diff(chosen) == 1):
                    block = vectors[:, int(chosen[0]) : int(chosen[-1]) + 1]
                else:
                    block = vectors[:, cp.asarray(chosen)]
                contraction.add(parts.unfold_block(cp.ascontiguousarray(block)), weights[places[chosen]])
        return "sectors"
    occupied = np.flatnonzero(occupations != 0.0)
    if not occupied.size:
        return "none"
    last = int(occupied[-1]) + 1
    rows = int(spinors.shape[0])
    width = _block_width(rows)
    on_device = isinstance(spinors, cp.ndarray)
    for start in range(0, last, width):
        stop = min(last, start + width)
        block = spinors[:, start:stop]
        if not on_device:
            block = cp.asarray(np.ascontiguousarray(block), dtype=cp.complex128)
        contraction.add(block, occupations[start:stop])
    return "device" if on_device else "host"


def fortran_spin_orbit_operator(grid, atoms, potentials, specifications, projectors):
    """The spin-orbit projectors ``projectors`` with the gradients of the Fortran-derived routines
    (:func:`parsec_python.Forces.spin_orbit_fortran.build_spin_orbit_projector_gradients`) as the sparse matrices of
    :class:`parsec_python.Forces.spin_orbit.SpinOrbitForceOperator`, or ``None`` without spin-orbit atoms."""

    from parsec_python.Forces.spin_orbit import SpinOrbitForceOperator, _stack
    from parsec_python.Forces.spin_orbit_fortran import build_spin_orbit_projector_gradients

    if not projectors:
        return None
    built = build_spin_orbit_projector_gradients(grid, atoms, potentials, specifications)
    by_atom = {item.atom_index: item for item in built}
    overlap_blocks: list = []
    gradient_blocks: list = [[], [], []]
    signs = np.zeros((len(projectors), 8))
    for index, atom in enumerate(projectors):
        gradient = by_atom[atom.atom_index]
        if not np.array_equal(gradient.support_rows, atom.support_rows):
            raise ValueError("spin-orbit projectors and gradients do not share their rows")
        rows = atom.support_rows
        columns = np.tile(rows, 16)
        row_index = np.repeat(np.arange(16), rows.size)

        def block(values: np.ndarray) -> sp.csr_matrix:  # values: (16, rows)
            return sp.csr_matrix((values.ravel(), (row_index, columns)), shape=(16, grid.size), dtype=complex)

        overlap_blocks.append(block(atom.stacked_conj_transpose))
        for axis in range(3):
            gradient_blocks[axis].append(block(gradient.gradients[axis].conj().T))
        signs[index, :3] = atom.sign[0]
        signs[index, 3:] = atom.sign[1]
    owners = np.fromiter((atom.atom_index for atom in projectors), dtype=np.int64, count=len(projectors))
    return SpinOrbitForceOperator(
        overlap=_stack(overlap_blocks, grid.size),
        gradient=tuple(_stack(blocks, grid.size) for blocks in gradient_blocks),  # type: ignore[arg-type]
        signs=signs,
        owners=owners,
    )


def compute_device_spinor_forces(
    system,
    result,
    *,
    symmetrize: bool = False,
    operations: Sequence | None = None,
    fortran: bool = False,
):
    """Forces of a finished accelerated self-consistent spin-orbit loop of a cluster.

    ``system`` is the prepared accelerated system (CuPy) and ``result`` the
    :class:`parsec_python.models.SelfConsistentSOCResult` of :func:`..SCF.self_consistent_soc.run_self_consistent_soc`
    with ``keep_device_spinors``: the spinors it kept on the device (``result.device_spinors``) are read where they
    are and released afterwards; without them the host spinors of the result are uploaded a block at a time.  The terms,
    symmetrization (``symmetrize``, ``operations``) and net-force removal are those of
    :func:`parsec_python.Forces.compute_spinor_forces` (the default routines) or, with ``fortran``, of
    :func:`parsec_python.Forces.spin_orbit_fortran.compute_spin_orbit_forces` (the Fortran-derived routines: their
    spin-orbit projector gradients, the spin-orbit term apart from the nonlocal one and the torque of the symmetrized
    forces kept); ``phase_seconds`` gives the seconds of the local and nonlocal terms and, as the code
    ``nonlocal_spinors_read``, where the spinors were read (1 the device sectors, 2 device whole-grid spinors, 3 the
    host).
    """

    from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
    from parsec_python.Forces import (
        assemble_forces,
        atom_symmetry_operations,
        build_nonlocal_projector_gradients,
        ion_ion_forces,
        local_forces,
    )
    from parsec_python.Forces.spin_orbit import build_spin_orbit_force_operator
    from parsec_python.Hartree.variational import hartree_tail_forces
    from parsec_python.Pseudopotential.radial_derivatives import linear_slope_setting

    from ..backends.cupy import require_cupy
    from .compute import correction_forces
    from .local_device import cupy_local_forces

    for name in ("xc_potential", "input_effective_potential", "output_effective_potential"):
        if getattr(result, name, None) is None:
            raise ValueError(f"the result does not carry {name}: it is not that of a spinor SCF driver of this package")
    cp, _ = require_cupy()
    # the device of the spinor loop (it solves on one), else the current one
    kept = getattr(result, "device_spinors_device", None)
    device = int(cp.cuda.Device().id) if kept is None else int(kept)
    started = perf_counter()
    clock: dict[str, float] = {}
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    repulsion = ion_ion_forces(atoms, system.pseudopotentials)
    clock["ion_ion"] = perf_counter() - started

    mark = perf_counter()
    if fortran:
        from parsec_python.Forces.spin_orbit_fortran import spinor_force_fields

        density, residual, xc = spinor_force_fields(result)
    else:
        density = result.density
        residual = np.asarray(result.output_effective_potential) - np.asarray(result.input_effective_potential)
        xc = result.xc_potential
    fields = (system.grid, atoms, system.pseudopotentials, specifications, density, residual, xc)
    # PARSEC_LINEAR_TABLE_SLOPES=interpolant with a species of linear tables: the readable sums, as on the scalar route
    host_slopes = linear_slope_setting() == "interpolant" and any(
        not specifications[atom.symbol].use_spline for atom in atoms
    )
    if host_slopes:
        local = local_forces(*fields)
    else:
        local = cupy_local_forces(*fields, device_ids=(device,))
    clock["local"] = perf_counter() - mark

    mark = perf_counter()
    gradients = build_nonlocal_projector_gradients(system.grid, atoms, system.pseudopotentials, specifications)
    projectors = getattr(result, "soc_projectors", None)
    if projectors is None:
        projectors = build_spin_orbit_projectors(system.grid, atoms, system.pseudopotentials, specifications)
    if fortran:
        spin_orbit = fortran_spin_orbit_operator(system.grid, atoms, system.pseudopotentials, specifications, projectors)
    else:
        spin_orbit = (
            build_spin_orbit_force_operator(system.grid, atoms, system.pseudopotentials, specifications, projectors)
            if projectors
            else None
        )
    clock["nonlocal_setup"] = perf_counter() - mark
    mark = perf_counter()
    spinors = getattr(result, "device_spinors", None)
    if spinors is None:
        spinors = result.spinors
    with cp.cuda.Device(device):
        contraction = SpinorForceContraction(cp, system.nonlocal_operator, gradients, spin_orbit)
        route = contract_spinors(contraction, spinors, np.asarray(result.occupations, dtype=np.float64))
        separable, spin_orbit_term = contraction.parts(len(atoms))
        del contraction
    release_device_spinors(result)
    clock["nonlocal"] = perf_counter() - mark

    mark = perf_counter()
    corrections = correction_forces(system, result)
    if corrections:
        clock["corrections"] = perf_counter() - mark
    tail_term = hartree_tail_forces(system, result)
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = atom_symmetry_operations(atoms)
    forces = assemble_forces(
        positions,
        repulsion,
        local,
        separable if fortran else separable + spin_orbit_term,
        operations,
        phase_seconds=clock,
        # the Fortran-derived routines report the spin-orbit term apart and keep the torque (forcso of soforc.F90)
        **({"keep_torque": True, "spin_orbit": spin_orbit_term} if fortran else {}),
        **corrections,
        **({} if tail_term is None else {"hartree_boundary": tail_term}),
    )
    # where the spinors were read (a code, not seconds): 1 the device sectors, 2 device whole-grid spinors, 3 the host
    clock["nonlocal_spinors_read"] = float({"sectors": 1, "device": 2, "host": 3}.get(route, 0))
    return replace(forces, seconds=perf_counter() - started)


def release_device_spinors(result) -> None:
    """Drop the device spinors a result kept for its forces (the next structure's loop needs the memory)."""

    for name in ("device_spinors", "device_spinors_device"):
        if getattr(result, name, None) is not None:
            setattr(result, name, None)


__all__ = [
    "SWITCH",
    "SpinorForceContraction",
    "compute_device_spinor_forces",
    "contract_spinors",
    "fortran_spin_orbit_operator",
    "release_device_spinors",
    "soc_device_forces_setting",
]
