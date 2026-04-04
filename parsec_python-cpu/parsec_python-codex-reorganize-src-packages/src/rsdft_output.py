"""Output and reporting helpers for the refactored RSDFT workflow.

This module is intentionally side-effect oriented: it owns the run log,
density snapshots, timing summaries, and optional ``wfn.dat`` output so the
solver module can stay focused on numerical flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from rsdft_models import A0_ANG, EnergyComponents, RY_EV


@dataclass
class TimingRecorder:
    """Collect setup-stage timings before flushing them to the log file."""
    entries: list[tuple[str, float]] = field(default_factory=list)

    def add(self, label: str, elapsed_seconds: float) -> None:
        """Record one labeled setup-stage timing entry."""
        self.entries.append((label, elapsed_seconds))

    def flush(self, filename: str) -> None:
        """Append all recorded timings to ``filename`` and clear the buffer."""
        if not self.entries:
            return
        with open(filename, "a", encoding="utf-8") as fid:
            fid.write("\n Setup timings [sec]\n")
            fid.write(" --------------------------------------------------\n")
            for label, elapsed_seconds in self.entries:
                fid.write(f" {label:<35} {elapsed_seconds:12.6f}\n")
        self.entries.clear()


def initialize_output_file(output_file: str, backend_label: str, density_method: str) -> None:
    """Create/overwrite the main text output file and write the run header."""
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fid:
        fid.write(f"Solver backend: {backend_label}\n")
        fid.write(f"Density initialization: {density_method}\n")


def save_density_variants(density_grid, domain, base_path: str) -> None:
    """Save density on the native grid plus Bohr^-3 and Angstrom^-3 variants."""
    if density_grid is None:
        return

    flat = np.asarray(density_grid).reshape(-1)
    nx = int(domain.get("nx", 0))
    ny = int(domain.get("ny", 0))
    nz = int(domain.get("nz", 0))
    expected = nx * ny * nz
    if expected <= 0 or flat.size != expected:
        print(f"Warning: density size {flat.size} does not match grid {expected}, skipping {base_path}")
        return

    h = float(domain.get("h", 0.0))
    if h <= 0:
        print(f"Warning: invalid grid spacing {h}, skipping {base_path}")
        return

    density_grid_3d = flat.reshape((nx, ny, nz), order="F")
    density_bohr3_3d = density_grid_3d / (h**3)
    density_ang3_3d = density_bohr3_3d / (A0_ANG**3)

    np.save(f"{base_path}_grid.npy", density_grid_3d)
    np.save(f"{base_path}_bohr3.npy", density_bohr3_3d)
    np.save(f"{base_path}.npy", density_ang3_3d)

    electron_grid = float(density_grid_3d.sum())
    electron_bohr3 = float(density_bohr3_3d.sum() * (h**3))
    h_ang = h * A0_ANG
    electron_ang3 = float(density_ang3_3d.sum() * (h_ang**3))
    print(
        f"Saved density variants for {os.path.basename(base_path)}: "
        f"Ne(grid)={electron_grid:.6f}, Ne(bohr^3)={electron_bohr3:.6f}, Ne(ang^3)={electron_ang3:.6f}"
    )


def write_rsdft_parameter_output(
    filename: str,
    nev: int,
    atoms,
    n_atom,
    domain,
    h: float,
    poldeg: int,
    fd_order: int,
) -> None:
    """Append geometry/domain metadata to the text log.

    Inputs are the derived solver problem quantities, and the output is a
    human-readable block in ``filename`` describing the run setup.
    """
    with open(filename, "a", encoding="utf-8") as fid:
        fid.write("\n")
        fid.write(f" Number of states: \t{nev}\n\n")
        fid.write("Atom data:\n -------------\n")
        fid.write(f" Total # of atom types is {len(atoms)}\n")

        atom_count = 0
        for at_typ, atom in enumerate(atoms):
            xyz = atom["coord"]
            fid.write(f" There are {n_atom[at_typ]} {atom['typ']} atoms\n")
            fid.write(" and their coordinates are:\n\n")
            fid.write("\t{:<12s}\t{:<12s}\t{:<12s}\n".format("x [a.u.]", "y [a.u.]", "z [a.u.]"))

            atom_count += n_atom[at_typ]
            for index in range(n_atom[at_typ]):
                fid.write(f"\t{xyz[index, 0]:12.6f}\t{xyz[index, 1]:12.6f}\t{xyz[index, 2]:12.6f}\n")
            fid.write("\n")

            xyz_ang = xyz * A0_ANG
            fid.write("\t{:<12s}\t{:<12s}\t{:<12s}\n".format("x [Ang]", "y [Ang]", "z [Ang]"))
            for index in range(n_atom[at_typ]):
                fid.write(
                    f"\t{xyz_ang[index, 0]:12.6f}\t{xyz_ang[index, 1]:12.6f}\t{xyz_ang[index, 2]:12.6f}\n"
                )
            fid.write("\n")

        fid.write(" --------------------------------------------------\n")
        fid.write(f" Total number of atoms :         {atom_count}\n\n")
        fid.write(f" Number of states:               {nev:10d} \n")
        fid.write(f" h grid spacing :                {h:10.5f} a.u.   ({h*A0_ANG:10.5f} Ang)\n")
        fid.write(f' Hamiltonian size :              {domain["nx"] * domain["ny"] * domain["nz"]:10d}  \n')
        fid.write(
            f' Sphere Radius :                 {domain["radius"]:10.5f} a.u.   '
            f'({domain["radius"]*A0_ANG:10.5f} Ang)\n'
        )
        fid.write(f' # grid points in each direction {domain["nx"]:10d}  \n')
        fid.write(f" Polynomial degree used :        {poldeg:10d}  \n")
        fid.write(f" Finite difference order :       {fd_order:10d}  \n")
        fid.write(" --------------------------------------------------\n")


def write_diag_info(output_file: str, diag_info: dict | None) -> None:
    """Write optional diagnostics returned by the initial-density builder."""
    if not diag_info:
        return

    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write(" --------------------------------------------------\n")
        fid.write(" Initial density electron count\n")
        if diag_info.get("electron_count_initial") is not None:
            fid.write(f" Initial density integrates to :     {diag_info['electron_count_initial']:.6f} e-\n")
        if diag_info.get("electron_count_normalized") is not None:
            if diag_info.get("electron_target") is not None:
                fid.write(
                    " Normalized density integrates to:  "
                    f"{diag_info['electron_count_normalized']:.6f} e- "
                    f"(target {diag_info['electron_target']:.6f})\n"
                )
            else:
                fid.write(
                    f" Normalized density integrates to:  {diag_info['electron_count_normalized']:.6f} e-\n"
                )
        fid.write(" --------------------------------------------------\n")


def write_initial_density_diagnostics(output_file: str, rhoxc, hpsum0_ev: float, exc_ry: float) -> None:
    """Log basic diagnostics for the starting density and XC/Hartree terms."""
    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write(
            f" max and min values of charge density [e/bohr^3]   {np.max(rhoxc):.5e}   {np.min(rhoxc):.5e}\n"
        )
        fid.write(f" Initial Hartree energy (eV) = {hpsum0_ev:10.5f}  \n")
        fid.write(f" Initial Exchange-corr. energy (eV) = {exc_ry * RY_EV:10.5f}  \n")


def write_scf_iteration(
    output_file: str,
    iteration: int,
    diag_label: str,
    diag_time: float,
    nev: int,
    lam_host,
    occup_host,
    hart_tol: float,
    hart_prec_label: str,
    hart_time: float,
    error: float,
    energies: EnergyComponents,
    n_atoms: int,
) -> None:
    """Append one SCF-iteration summary block to the text log."""
    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write(f"\n\n SCF iter # {iteration}  ... \n")
        fid.write(f"Diagonalization method :\t{diag_label}\n")
        fid.write(f"Diagonalization time [sec] :\t{diag_time}\n\n")
        fid.write("   State  Eigenvalue [Ry]     Eigenvalue [eV]  Occupation\n\n")
        for index in range(nev):
            eig_ev = float(lam_host[index]) * 2 * RY_EV
            eig_ry = eig_ev / RY_EV
            occ = 2 * float(occup_host[index])
            fid.write(f"{index + 1:5d}   {eig_ry:15.10f}   {eig_ev:18.10f}  {occ:5.2f}\n")

        fid.write(f"Hartree CG tol :\t{hart_tol:.1e} ({hart_prec_label})\n")
        fid.write(f"\nHartree potential time [sec]: \t{hart_time}\n\n")
        fid.write(f"   ... SCF error = {error:10.2e}\n")
        fid.write("   Energy components this iter:\n")
        fid.write(
            f"     Sum of eigenvalues      = {energies.eigen_sum_ry * RY_EV:10.5f}  eV   = "
            f"{energies.eigen_sum_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f"     Hartree energy          = {energies.hartree_ry * RY_EV:10.5f}  eV   = "
            f"{energies.hartree_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f"     Exchange-corr. energy   = {energies.xc_ry * RY_EV:10.5f}  eV   = "
            f"{energies.xc_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f"     Ion-ion repulsion       = {energies.ion_ry * RY_EV:10.5f}  eV   = "
            f"{energies.ion_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f"     Total electronic energy = {energies.total_ry * RY_EV:10.5f}  eV   = "
            f"{energies.total_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f"     Electronic energy/atom  = {(energies.total_ry * RY_EV) / n_atoms:10.5f}  eV   = "
            f"{energies.total_ry / n_atoms:10.5f}  Ry  \n"
        )


def print_eigenvalues_to_console(nev: int, lam_host, occup_host) -> None:
    """Print the final occupied-state eigenvalues and occupations to stdout."""
    print("   State  Eigenvalue [Ry]     Eigenvalue [eV]  Occupation ")
    for index in range(nev):
        eig_ev = float(lam_host[index]) * 2 * RY_EV
        eig_ry = eig_ev / RY_EV
        occ = 2 * float(occup_host[index])
        print(f"{index + 1:5d}   {eig_ry:15.4f}   {eig_ev:18.3f}  {occ:10.2f}")


def write_total_energy_summary(output_file: str, energies: EnergyComponents, n_atoms: int) -> None:
    """Append the final total-energy summary to the output file."""
    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write("\n\n")
        fid.write(" Total Energies \n\n")
        fid.write(
            f" Sum of eigenvalues      = {energies.eigen_sum_ry * RY_EV:10.5f}  eV   = "
            f"{energies.eigen_sum_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f" Hartree energy          = {energies.hartree_ry * RY_EV:10.5f}  eV   = "
            f"{energies.hartree_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f" Exchange-corr. energy   = {energies.xc_ry * RY_EV:10.5f}  eV   = "
            f"{energies.xc_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f" Ion-ion repulsion       = {energies.ion_ry * RY_EV:10.5f}  eV   = "
            f"{energies.ion_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f" Total electronic energy = {energies.total_ry * RY_EV:10.5f}  eV   = "
            f"{energies.total_ry:10.5f}  Ry  \n"
        )
        fid.write(
            f" Electronic energy/atom  = {(energies.total_ry * RY_EV) / n_atoms:10.5f}  eV   = "
            f"{energies.total_ry / n_atoms:10.5f}  Ry  \n"
        )


def print_total_energy_summary(energies: EnergyComponents, n_atoms: int) -> None:
    """Print the final total-energy summary to stdout."""
    print("\n Total Energies \n\n")
    print(
        f" Sum of eigenvalues      = {energies.eigen_sum_ry * RY_EV:10.5f}  eV   = {energies.eigen_sum_ry:10.4f}  Ry  "
    )
    print(
        f" Hartree energy          = {energies.hartree_ry * RY_EV:10.5f}  eV   = {energies.hartree_ry:10.4f}  Ry  "
    )
    print(
        f" Exchange-corr. energy   = {energies.xc_ry * RY_EV:10.5f}  eV   = {energies.xc_ry:10.4f}  Ry  "
    )
    print(
        f" Ion-ion repulsion       = {energies.ion_ry * RY_EV:10.5f}  eV   = {energies.ion_ry:10.4f}  Ry  "
    )
    print(
        f" Total electronic energy = {energies.total_ry * RY_EV:10.5f}  eV   = {energies.total_ry:10.4f}  Ry  "
    )
    print(
        f" Electronic energy/atom  = {(energies.total_ry * RY_EV) / n_atoms:10.5f}  eV   = "
        f"{energies.total_ry / n_atoms:10.4f}  Ry  "
    )


def save_wavefunction(
    wfn_file: str,
    domain,
    potential,
    rho,
    wavefunctions,
    nev: int,
    n_types: int,
    atoms,
    to_numpy_array,
) -> None:
    """Write the legacy ``wfn.dat`` binary output for restart/post-processing."""
    with open(wfn_file, "wb") as wfnid:
        wfnid.write(np.array(26, dtype=np.uint32).tobytes())
        wfnid.write(np.array(domain["radius"], dtype=np.float64).tobytes())
        wfnid.write(np.array(domain["h"], dtype=np.float64).tobytes())

        pot_host = np.asarray(potential, dtype=np.float64)
        wfnid.write(np.array(len(pot_host), dtype=np.uint32).tobytes())
        wfnid.write(pot_host.tobytes())

        rho_host = np.asarray(rho, dtype=np.float64)
        wfnid.write(np.array(len(rho_host), dtype=np.uint32).tobytes())
        wfnid.write(rho_host.tobytes())

        w_host = np.asarray(to_numpy_array(wavefunctions), dtype=np.float64)
        wfnid.write(np.array(len(w_host), dtype=np.uint32).tobytes())
        wfnid.write(np.array(nev, dtype=np.uint32).tobytes())
        for index in range(nev):
            wfnid.write(np.array(w_host[:, index], dtype=np.float64).tobytes())

        wfnid.write(np.array(n_types, dtype=np.uint32).tobytes())
        for atom in atoms:
            xyz = atom["coord"]
            wfnid.write(np.array(len(xyz), dtype=np.uint32).tobytes())
            for row in range(len(xyz)):
                wfnid.write(np.array(xyz[row, :], dtype=np.float64).tobytes())
