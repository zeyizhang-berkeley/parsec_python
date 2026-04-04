"""Numerical RSDFT driver for the refactored code path.

The main public function here is ``run_rsdft_calculation()``. It performs the
same broad stages as the legacy ``main.py``:
1. Build the Laplacian and ionic terms.
2. Construct the initial density/potentials.
3. Build nonlocal and XC pieces.
4. Run the self-consistent-field loop.
5. Save final densities, energies, and optional wavefunctions.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Eigensolvers.pcg import pcg as pcg_cpu
from Laplacian.fd3d import fd3d
from Laplacian.nuclear import nuclear
from Mixer.mixer import mixer, reset_mixer
from V_ion.nelectrons import nelectrons
from V_ion.pseudoNL_original_ML4Den import pseudoNL_ML4Den
from V_xc.exc_nspn import exc_nspn

from rsdft_models import EnergyComponents, PreparedSystem, SCFResult, SolverBackend
from rsdft_output import (
    TimingRecorder,
    print_eigenvalues_to_console,
    print_total_energy_summary,
    save_density_variants,
    save_wavefunction,
    write_diag_info,
    write_initial_density_diagnostics,
    write_scf_iteration,
    write_total_energy_summary,
)


def _scaled_poldeg(poldeg: int, degree_modifier: float) -> int:
    """Convert the adaptive polynomial scale factor into a safe integer degree."""
    return max(1, int(round(poldeg * degree_modifier)))


def _scaled_lanczos_steps(nev: int, modifier: float) -> int:
    """Convert the adaptive Lanczos subspace scaling into an integer step count."""
    return nev + max(1, int(round(500 * modifier)))


def _build_preconditioner(A, cg_prec: int):
    """Build the ILU preconditioner used by the CPU Hartree solve."""
    if not cg_prec:
        return []

    print("Calling ilu0 ...")
    preconditioner = spla.spilu(A)
    print("done.")
    return preconditioner


def _random_lanczos_vector(backend: SolverBackend, n: int):
    """Create the initial random vector for Lanczos on the active backend."""
    if backend.label == "gpu":
        cp = backend.cupy_module
        return cp.random.randn(n, 1, dtype=cp.float32)
    return np.random.randn(n, 1)


def _compute_initial_density(
    problem: PreparedSystem,
    elem,
    n_elements: int,
    backend: SolverBackend,
    A,
    preconditioner,
):
    """Build the initial density and diagonal ionic potentials.

    Output:
        ``(rho0, hpot0, ppot, diag_info)`` matching the legacy main script.
    """
    density_method = problem.input_data.density_method
    if density_method == "sad":
        print("Using SAD method...")
        return backend.pseudo_diag(problem.domain, problem.input_data.atoms, elem, n_elements, return_info=True)

    if density_method == "sad_ml_grid":
        print(f"Using SAD density on ML grid from: {problem.input_data.grid_npy_path}")
        from V_ion.pseudoDiag_MLgrid import pseudoDiag_MLgrid

        return pseudoDiag_MLgrid(problem.domain, problem.input_data.atoms, elem, n_elements, return_info=True)

    print(f"Using ML grid/density file: {problem.input_data.ml_file_path}")
    from V_ion.pseudoDiag_ML4Den_poisson import pseudoDiag_ML4Den

    return pseudoDiag_ML4Den(
        problem.domain,
        problem.input_data.atoms,
        elem,
        n_elements,
        problem.input_data.ml_file_path,
        A,
        problem.settings.cg_prec,
        preconditioner,
        return_info=True,
    )


def _compute_nonlocal_potential(problem: PreparedSystem, elem, n_elements: int, backend: SolverBackend):
    """Build the nonlocal pseudopotential operator for the chosen density mode."""
    density_method = problem.input_data.density_method
    if density_method == "sad":
        print("Using SAD method...")
        return backend.pseudo_nl(problem.domain, problem.input_data.atoms, elem, n_elements)

    if density_method == "sad_ml_grid":
        print(f"Using SAD density on ML grid from: {problem.input_data.grid_npy_path}")
        from V_ion.pseudoNL_original_MLgrid import pseudoNL_MLgrid

        return pseudoNL_MLgrid(problem.domain, problem.input_data.atoms, elem, n_elements)

    print(f"Using ML grid/density file: {problem.input_data.ml_file_path}")
    return pseudoNL_ML4Den(problem.domain, problem.input_data.atoms, elem, n_elements)


def compute_energy_components(
    rho,
    hpot,
    hpot0,
    xc_potential,
    exc: float,
    lam_host,
    occup_host,
    nev: int,
    e_nuc0: float,
    h: float,
) -> EnergyComponents:
    """Package the current energy breakdown into an ``EnergyComponents`` object."""
    esum = float(np.sum(lam_host[:nev] * occup_host[:nev]))
    eigen_sum_ry = 4.0 * esum
    hartree_ry = float(np.sum(rho * (hpot + hpot0)) * h**3)
    vxc_sum_ry = float(np.sum(rho * xc_potential) * h**3)
    xc_ry = float(exc)
    total_ry = eigen_sum_ry - hartree_ry + xc_ry - vxc_sum_ry + float(e_nuc0)
    return EnergyComponents(
        eigen_sum_ry=eigen_sum_ry,
        hartree_ry=hartree_ry,
        xc_ry=xc_ry,
        ion_ry=float(e_nuc0),
        total_ry=total_ry,
    )


def run_rsdft_calculation(
    problem: PreparedSystem,
    elem,
    n_elements: int,
    backend: SolverBackend,
) -> SCFResult:
    """Execute the real-space DFT solve for one prepared system.

    Input:
        problem: fully prepared run data from ``rsdft_setup.prepare_system``.
        elem, n_elements: element lookup table and its size.
        backend: CPU/GPU implementation bundle selected earlier.

    Output:
        ``SCFResult`` containing the final density, potentials, eigenpairs,
        convergence flag, and other end-of-run quantities.
    """
    print(" ")
    print("******************")
    print("     OUTPUT       ")
    print("******************")
    print(" ")
    print(" Working.....constructing Laplacian matrix...")

    timings = TimingRecorder()
    h = problem.domain["h"]
    nx, ny, nz = problem.domain["nx"], problem.domain["ny"], problem.domain["nz"]

    # Stage 1: build the finite-difference Laplacian on the chosen domain.
    start_time = time.time()
    A = (1.0 / (h * h)) * fd3d(nx, ny, nz, problem.settings.fd_order)
    laplacian_time = time.time() - start_time
    print(laplacian_time)
    timings.add("Laplacian construction", laplacian_time)

    n = A.shape[0]
    Hpot = np.zeros(n)
    pot = Hpot.copy()
    err = 10.0 + problem.settings.tol

    # Stage 2: set up ionic terms and the initial charge density.
    print(" Working.....setting up ionic potential...")
    start_time = time.time()
    e_nuc0 = nuclear(problem.domain, problem.input_data.atoms, elem, n_elements)
    enuc_time = time.time() - start_time
    print(" Enuc time", enuc_time)
    timings.add("Ion-ion repulsion setup", enuc_time)

    preconditioner = _build_preconditioner(A, problem.settings.cg_prec)

    print(" Working.....setting up diagonal part of ionic potential...")
    start_time = time.time()
    rho0, hpot0, ppot, diag_info = _compute_initial_density(
        problem,
        elem,
        n_elements,
        backend,
        A,
        preconditioner,
    )
    rho0 = np.asarray(backend.to_numpy_array(rho0)).reshape(-1)
    hpot0 = np.asarray(backend.to_numpy_array(hpot0)).reshape(-1)
    ppot = np.asarray(backend.to_numpy_array(ppot)).reshape(-1)
    pseudo_diag_time = time.time() - start_time
    print(" pseudoDiag time: ", pseudo_diag_time)
    timings.add("Diagonal ionic potential setup", pseudo_diag_time)
    write_diag_info(problem.paths.output_file, diag_info)

    if problem.input_data.z_charge != 0:
        scaling_factor = problem.ztest / problem.zelec
        rho0 *= scaling_factor
        hpot0 *= scaling_factor

    save_density_variants(rho0, problem.domain, problem.paths.initial_density_base)

    rhoxc = np.transpose(rho0) / (h**3)
    hpsum0 = float(np.sum(rho0 * hpot0))
    hpsum0_ev = hpsum0 * 13.605698066

    print(" Working.....setting up nonlocal part of ionic potential...")
    start_time = time.time()
    vnl = _compute_nonlocal_potential(problem, elem, n_elements, backend)
    pseudo_nl_time = time.time() - start_time
    print(" pseudoNL time: ", pseudo_nl_time)
    timings.add("Nonlocal ionic potential setup", pseudo_nl_time)

    print(" Working.....setting up exchange and correlation potentials...")
    start_time = time.time()
    XCpot, exc = exc_nspn(problem.domain, rhoxc)
    exc_time = time.time() - start_time
    print(" exc time: ", exc_time)
    timings.add("Exchange-correlation setup", exc_time)

    write_initial_density_diagnostics(problem.paths.output_file, rhoxc, hpsum0_ev, exc)
    timings.flush(problem.paths.output_file)

    xcpot = np.transpose(XCpot)
    nelec = nelectrons(problem.input_data.atoms, elem, n_elements)
    if problem.input_data.z_charge != 0:
        nelec -= problem.input_data.z_charge

    pot = ppot + hpot0 + 0.5 * xcpot
    reset_mixer()

    with open(problem.paths.output_file, "a", encoding="utf-8") as fid:
        fid.write("\n----------------------------------\n\n")

    half_a_plus_vnl = 0.5 * A + vnl
    if problem.settings.adaptive_scheme != 0 and sum(problem.input_data.n_atom) <= 2:
        degree_modifier = 0.75
        m_modifier = 0.95
    else:
        degree_modifier = 1.0
        m_modifier = 1.0

    W = []
    lam = []
    occup = []
    rho = np.zeros(n)
    lam_host = np.array([])
    occup_host = np.array([])
    n_atoms = sum(atom["coord"].shape[0] for atom in problem.input_data.atoms)
    iterations = 0

    # Stage 3: self-consistent-field loop.
    while err > problem.settings.tol and iterations < problem.settings.maxits:
        iterations += 1
        print(f"  Working ... SCF iter # {iterations} ... ")

        B = half_a_plus_vnl + sp.diags(pot, 0, shape=(n, n))
        start_time = time.time()

        # Solve the current Hamiltonian using the selected eigensolver path.
        if problem.settings.diagmeth == 1 or (iterations == 1 and problem.settings.diagmeth == 0):
            print("Calling lanczos...")
            diag_label = f"lanczos_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.lanczos(
                B,
                problem.nev + 15,
                _random_lanczos_vector(backend, n),
                _scaled_lanczos_steps(problem.nev, m_modifier),
                1e-5,
            )
        elif iterations == 1 and problem.settings.diagmeth == 2:
            print("Calling chsubsp...")
            diag_label = f"chsubsp_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.chsubsp(_scaled_poldeg(problem.settings.poldeg, degree_modifier), problem.nev + 15, B)
        elif iterations == 1 and problem.settings.diagmeth == 3:
            print("Calling first_filt...")
            diag_label = f"first_filt_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.first_filt(problem.nev + 15, B, _scaled_poldeg(problem.settings.poldeg, degree_modifier))
        else:
            print("Calling chebsf...")
            diag_label = f"chebsf_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.chefsi1(
                W,
                lam,
                _scaled_poldeg(problem.settings.poldeg, degree_modifier),
                problem.nev,
                B,
            )

        diag_time = time.time() - start_time
        _, occup = backend.occupations(lam[: problem.nev], problem.settings.fermi_temp, nelec, 1e-6)
        lam_host = np.asarray(backend.to_numpy_array(lam)).reshape(-1)
        occup_host = np.asarray(backend.to_numpy_array(occup)).reshape(-1)

        # Update the density from the occupied states, then solve Poisson/XC.
        rho = (W[:, : problem.nev] ** 2) @ (2 * occup)
        rho = np.asarray(backend.to_numpy_array(rho)).reshape(-1)

        hrhs = (4 * np.pi / h**3) * (rho - rho0)
        rho = rho / h**3

        start_time = time.time()
        hart_tol = 1e-5
        if backend.label == "gpu" and problem.settings.cg_prec:
            hart_prec_label = "gpu-no-prec (ILU unavailable on GPU path)"
        else:
            hart_prec_label = "precLU" if problem.settings.cg_prec else "no prec"

        if problem.settings.cg_prec:
            print(f"with CG_prec (Hartree CG tol = {hart_tol:.1e})")
            if backend.label == "cpu":
                Hpot, _ = pcg_cpu(A, hrhs, Hpot, 200, hart_tol, preconditioner, "precLU")
            else:
                Hpot, _ = backend.pcg(A, hrhs, Hpot, 200, hart_tol, preconditioner, "precLU")
        else:
            print(f"no CG_prec (Hartree CG tol = {hart_tol:.1e})")
            Hpot, _ = backend.pcg(A, hrhs, Hpot, 200, hart_tol)

        Hpot = np.asarray(backend.to_numpy_array(Hpot)).reshape(-1)
        hart_time = time.time() - start_time

        XCpot, exc = exc_nspn(problem.domain, rho, problem.paths.output_file)
        pot_new = ppot + 0.5 * XCpot + Hpot + hpot0
        err_new = np.linalg.norm(pot_new - pot) / np.linalg.norm(pot_new)

        # Mild adaptive tuning of polynomial degree / Lanczos subspace size.
        if problem.settings.adaptive_scheme == 0 or err_new > 1 or err_new > 2 * err:
            degree_modifier = 1.0
            m_modifier = 1.0
        elif err_new > err:
            degree_modifier = min(1.1, degree_modifier + 0.2)
            m_modifier = min(1.1, degree_modifier + 0.05)
        elif 3 * err_new < err:
            degree_modifier = max(0.5, degree_modifier - 0.1)
            m_modifier = max(0.9, degree_modifier - 0.025)

        err = err_new
        iteration_energies = compute_energy_components(
            rho,
            Hpot,
            hpot0,
            XCpot,
            exc,
            lam_host,
            occup_host,
            problem.nev,
            e_nuc0,
            h,
        )
        write_scf_iteration(
            problem.paths.output_file,
            iterations,
            diag_label,
            diag_time,
            problem.nev,
            lam_host,
            occup_host,
            hart_tol,
            hart_prec_label,
            hart_time,
            err,
            iteration_energies,
            n_atoms,
        )
        print(f"   ... SCF error = {err:10.2e}\n")

        pot, _ = mixer(pot, pot_new - pot)

    # Stage 4: final reporting and optional wavefunction export.
    print("SCF loop completed.")
    converged = err <= problem.settings.tol
    if converged:
        print("          ")
        print("**************************")
        print(" CONVERGED SOLUTION!! ")
        print("**************************")
        print("         ")
    else:
        print("          ")
        print("**************************")
        print(" !!THE SYSTEM DID NOT CONVERGE!!")
        print("          ")
        print(" !!THESE ARE THE VALUES FROM THE LAST ITERATION!!")
        print("**************************")
        print("         ")

    save_density_variants(rho * (problem.domain["h"] ** 3), problem.domain, problem.paths.converged_density_base)
    print_eigenvalues_to_console(problem.nev, lam_host, occup_host)

    final_energies = compute_energy_components(
        rho,
        Hpot,
        hpot0,
        XCpot,
        exc,
        lam_host,
        occup_host,
        problem.nev,
        e_nuc0,
        h,
    )
    write_total_energy_summary(problem.paths.output_file, final_energies, n_atoms)
    print_total_energy_summary(final_energies, n_atoms)

    reset_mixer()
    if problem.settings.save_wfn:
        save_wavefunction(
            problem.paths.wfn_file,
            problem.domain,
            pot,
            rho,
            W,
            problem.nev,
            problem.n_types,
            problem.input_data.atoms,
            backend.to_numpy_array,
        )

    return SCFResult(
        rho=rho,
        hpot=Hpot,
        xc_potential=XCpot,
        exc=float(exc),
        potential=pot,
        wavefunctions=W,
        eigenvalues=lam_host,
        occupations=occup_host,
        iterations=iterations,
        error=err,
        converged=converged,
        e_nuc0=float(e_nuc0),
        hpot0=hpot0,
        n_atoms=n_atoms,
    )
