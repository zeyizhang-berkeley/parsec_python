import numpy as np
# from scipy.interpolate import RegularGridInterpolator # for linear interpolation
from scipy.ndimage import map_coordinates
from splineData import splineData
from preProcess import preProcess
from fspline import fspline
#import logging
from fsplevalIO import fsplevalIO
from pcg import pcg # Added pcg import

def pseudoDiag_ML4Den(Domain, Atoms, elem, N_elements, density_file_path, A, CG_prec=False, PRE=None):
    """
    Set up initial screening and local ionic pseudopotential using ML-predicted 3D density.
    This function replaces the traditional atomic density superposition approach with a machine learning predicted density read from a .npy file.
    Args:
        Domain: Grid domain information
        Atoms: Atomic coordinates and types
        elem: Element information
        N_elements: Number of elements
        density_file_path: Path to the .npy file containing ML-predicted 3D density data
        A: Discrete Laplacian matrix for Hartree potential calculation
        CG_prec: Whether to use preconditioned CG
        PRE: Preconditioner for CG (if CG_prec is True)

    Returns:
        rho0: ML-predicted initial charge density (flattened)
        hpot0: Hartree potential calculated from ML density
        pot: Pseudopotential from atomic superposition
    """
    OPTIMIZATIONLEVEL = 0
    enableMexFilesTest = 0

    # Extract domain information
    nx, ny, nz = Domain['nx'], Domain['ny'], Domain['nz']
    h = Domain['h']
    rad = Domain['radius']
    n = nx * ny * nz # Total number of grid points

    # Initialize output arrays
    rho0 = np.zeros(n) # Flattened for compatibility with SCF loop
    pot = np.zeros(n)
    hpot0 = np.zeros(n)

    # ==========================================
    # 1. PSEUDOPOTENTIAL CALCULATION (UNCHANGED FROM ATOMIC APPROACH)
    # ==========================================

    print("Calculating pseudopotential from atomic superposition...")
    # Get atomic functional data
    AtomFuncData, data_list = splineData()
    N_types = len(Atoms)
    Z_sum = 0.0

    # Atom-type loop
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']
        xyz = Atoms[at_typ]['coord']
        natoms = xyz.shape[0]

        # Look for matching element data in the elem DataFrame
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                Z = elem['Z'].iloc[i]
                Z_sum += Z * natoms

        # Search for the atom's data in AtomFuncData
        index = 0
        for i in range(len(AtomFuncData)):
            if typ == AtomFuncData[i]['atom']:
                index = i
                break

        # Localizing variables and initializing arrays
        # i_charge = data_list.index('charge')
        i_pot_S = data_list.index('pot_S')
        # i_hartree = data_list.index('hartree')

        atom_data = AtomFuncData[index]['data']
        # x_charg = atom_data[:, 0]
        # y_charg = atom_data[:, i_charge]
        x_pot_s = atom_data[:, 0]
        y_pot_s = atom_data[:, i_pot_S]
        # x_vhart = atom_data[:, 0]
        # y_vhart = atom_data[:, i_hartree]

        # Pre-processing the data
        # I = preProcess(y_charg)
        # x_charg, y_charg = x_charg[I], y_charg[I]

        I = preProcess(y_pot_s)
        x_pot_s, y_pot_s = x_pot_s[I], y_pot_s[I]

        # I = preProcess(y_vhart)
        # x_vhart, y_vhart = x_vhart[I], y_vhart[I]

        # Calculating the splines
        # z_chg, c_chg, d_chg = fspline(x_charg, y_charg)
        z_p_s, c_p_s, d_p_s = fspline(x_pot_s, y_pot_s)
        # z_vht, c_vht, d_vht = fspline(x_vhart, y_vhart)

        # Atom-specific computations (loop through the grid points)
        for at in range(natoms):
            indx = 0
            # Adjust coordinates to the grid
            k = np.arange(nz)
            dz = (k * h - rad - xyz[at, 2]) ** 2

            for k_idx in range(nz):
                j = np.arange(ny)
                dy = dz[k_idx] + (j * h - rad - xyz[at, 1]) ** 2

                for j_idx in range(ny):
                    i = np.arange(nx)
                    r1 = np.sqrt(dy[j_idx] + (i * h - rad - xyz[at, 0]) ** 2)

                    # Initialize arrays for potentials and charge
                    ppot = np.zeros(nx)
                    # rrho = np.zeros(nx)
                    # hpot00 = np.zeros(nx)

                    # Initialization of intervals
                    # j_ch = 0
                    j_p_s = 0
                    # j_vht = 0

                    # Check if optimization is enabled
                    if OPTIMIZATIONLEVEL != 0:
                        # TODO: switch to C++ code to compare the performance?
                        # Call the C function (PsuedoDiagLoops), or a Python placeholder
                        # ppot, rrho, hpot00 = PsuedoDiagLoops(
                        #     z_p_s, c_p_s, d_p_s, z_chg, c_chg, d_chg,
                        #     z_vht, c_vht, d_vht, x_pot_s, x_charg, x_vhart,
                        #     r1, j_p_s, j_ch, j_vht, nx
                        # )
                        ValueError("case not implemented: PsuedoDiagLoops")
                    else:
                        for i_idx in range(0, nx, 2):
                            iPlusOne = i_idx + 1
                            # Evaluate the splines
                            ppot[i_idx], j_p_s = fsplevalIO(z_p_s, c_p_s, d_p_s, x_pot_s, r1[i_idx], j_p_s)
                            # rrho[i_idx], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[i_idx], j_ch)
                            # hpot00[i_idx], j_vht = fsplevalIO(z_vht, c_vht, d_vht, x_vhart, r1[i_idx], j_vht)
                            ppot[iPlusOne], j_p_s = fsplevalIO(z_p_s, c_p_s, d_p_s, x_pot_s, r1[iPlusOne], j_p_s)
                            # rrho[iPlusOne], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[iPlusOne], j_ch)
                            # hpot00[iPlusOne], j_vht = fsplevalIO(z_vht, c_vht, d_vht, x_vhart, r1[iPlusOne], j_vht)
                        # done atom-specific calculations - now compute potentials, charge.

                    if enableMexFilesTest == 1:
                        ValueError("case not implemented: PsuedoDiagLoops")
                        # TODO: switch to C++ code to compare the performance?
                        # ppot2, rrho2, hpot002 = PsuedoDiagLoops(
                        #     z_p_s, c_p_s, d_p_s, z_chg, c_chg, d_chg,
                        #     z_vht, c_vht, d_vht, x_pot_s, x_charg, x_vhart,
                        #     r1, j_p_s, j_ch, j_vht, nx
                        # )
                        #
                        # # Check for discrepancies
                        # if (np.any(np.abs(ppot2 - ppot) > 1e-6) or
                        # np.any(np.abs(rrho2 - rrho) > 1e-6) or
                        # np.any(np.abs(hpot002 - hpot00) > 1e-6)):
                        # # Prepare the stack trace
                        # current_frame = inspect.currentframe()
                        # stack_trace = inspect.getouterframes(current_frame)
                        #
                        # # Raise custom exception with details
                        # exception = MexFileDiscrepancyError(
                            # message="Mex file discrepancy for PsuedoDiagLoops",
                            # identifier=None, # You can add an identifier if necessary
                            # stack=[frame.function for frame in stack_trace] # Extract function names from the stack
                        # )
                        # logError(exception)
                        # raise exception # Optionally raise the error after logging

                    # #rrho = np.maximum(0, rrho * (h ** 3) / (4 * np.pi))
                    indx_end = indx + nx

                    pot[indx:indx_end] += ppot
                    # rho0[indx:indx_end] += rrho
                    # hpot0[indx:indx_end] += hpot00

                    indx = indx_end



    # ========================================== #
    # 2. READ AND INTERPOLATE ML-PREDICTED 3D DENSITY
    # ==========================================

    print("Loading ML-predicted density from:", density_file_path)

    # Load the ML-predicted 3D density from .npy file
    density_3d = np.load(density_file_path)

    # 1. Get grid shape from ML density
    nx_file, ny_file, nz_file = density_3d.shape

    # 2. Set box length (in Bohr)
    box_length_bohr = 10 * 0.755890 # = 7.5589 Bohr

    # ✅ 3. Compute atomic center (in Bohr)
    geom_center = np.mean(np.array([atom['coord'] for atom in Atoms]), axis=0)

    # ✅ 4. Create and shift ML grid
    x_file = np.linspace(0, box_length_bohr, nx_file)
    y_file = np.linspace(0, box_length_bohr, ny_file)
    z_file = np.linspace(0, box_length_bohr, nz_file)

    x_file -= geom_center[0]
    y_file -= geom_center[1]
    z_file -= geom_center[2]

    # ✅ 5. Now print the correct range
    print("Target grid min/max:")
    print(f"ML X grid: {x_file.min():.2f} → {x_file.max():.2f}")

    # # Create the linear interpolator
    # density_interpolator = RegularGridInterpolator(
    #   (z_file, y_file, x_file),
    #   density_3d, # method='linear',
    #   bounds_error=False,
    #   fill_value=0.0
    # )

    # Create coordinate arrays for your target grid
    x_target = np.linspace(-rad, rad, nx)
    y_target = np.linspace(-rad, rad, ny)
    z_target = np.linspace(-rad, rad, nz)
    print(f"X: {x_target.min():.2f} → {x_target.max():.2f}")

    # Create meshgrid for interpolation
    X_target, Y_target, Z_target = np.meshgrid(x_target, y_target, z_target, indexing='ij')
    target_points = np.stack([X_target.ravel(), Y_target.ravel(), Z_target.ravel()], axis=-1)

    # Interpolate ML density to target grid and flatten
    print("Interpolating ML density to target grid...")

    # rho0_interpolated = density_interpolator(target_points)

    # Compute spacing of ML grid
    dx = x_file[1] - x_file[0]
    dy = y_file[1] - y_file[0]
    dz = z_file[1] - z_file[0]

    # Convert target physical coordinates → ML grid index coordinates
    Z_idx = (Z_target.ravel() - z_file[0]) / dz
    Y_idx = (Y_target.ravel() - y_file[0]) / dy
    X_idx = (X_target.ravel() - x_file[0]) / dx

    coords_idx = np.vstack([Z_idx, Y_idx, X_idx]) # shape: (3, N)

    # Perform cubic spline interpolation
    rho0_interpolated = map_coordinates(
        density_3d,
        coords_idx,
        order=3, # Cubic spline
        mode='constant',
        cval=0.0
    )

    rho0 = rho0_interpolated.reshape(n)

    print(f"▶ max density after spline interpolation: {rho0_interpolated.max():.6f}")

    idx = np.unravel_index(np.argmax(rho0), rho0.shape)
    print("Density peak index:", idx)
    print("Peak location (Bohr):", x_target[idx[0]], y_target[idx[1]], z_target[idx[2]])

    # assume rho0 is your raw interpolated array, and h is your DFT grid spacing
    # and Z_sum is your total number of electrons

    # 1. Integration on DFT grid BEFORE any unit conversion or clipping
    electron_count_raw = np.sum(rho0) * h ** 3
    print(f"▶ raw interpolated ML density integrates to {electron_count_raw:.6f} e⁻")

    # ==========================================
    # UNIT CONVERSION: Å³ → bohr³
    # ==========================================
    BOHR_TO_ANGSTROM = 0.529177210903
    ANGSTROM3_TO_BOHR3 = 1.0 / (BOHR_TO_ANGSTROM ** 3)

    print(f"ML density before unit conversion: min={np.min(rho0):.6f}, max={np.max(rho0):.6f} electrons/Å³")

    # Convert from electrons/Å³ to electrons/bohr³
    rho0 = rho0 * ANGSTROM3_TO_BOHR3

    print(f"ML density after unit conversion: min={np.min(rho0):.6f}, max={np.max(rho0):.6f} electrons/bohr³")
    print(f"Conversion factor: {ANGSTROM3_TO_BOHR3:.6f}")

    # 2. After any unit conversion (e.g. Å³→bohr³), but BEFORE normalization
    # (only if you actually convert; if you commented it out, skip this)
    electron_count_conv = np.sum(rho0) * h ** 3
    print(f"▶ post-conversion (if applied) integrates to {electron_count_conv:.6f} e⁻")

    # ==========================================
    # ENFORCE NON-NEGATIVE DENSITY (PHYSICAL CONSTRAINT)
    # ==========================================

    # Count negative values before correction
    negative_count = np.sum(rho0 < 0)
    if negative_count > 0:
        print(f"Warning: Found {negative_count} negative density values after interpolation")
        print(f"Min density before correction: {np.min(rho0):.6f}")

        # Set negative values to zero
        rho0[rho0 < 0] = 0.0
        print("Negative densities set to zero (physical constraint)")

    print(f"ML density statistics: min={np.min(rho0):.6f}, max={np.max(rho0):.6f}, mean={np.mean(rho0):.6f}")

    # Normalize rho0
    # rho0_sum = np.sum(rho0)
    # rho0 = Z_sum * rho0 / rho0_sum
    # correct normalization:
    vol = h ** 3
    int_rho = np.sum(rho0) * vol # electrons
    rho0 *= (Z_sum / int_rho)
    print(f"ML density statistics: min={np.min(rho0):.6f}, max={np.max(rho0):.6f}, mean={np.mean(rho0):.6f}")

    # 3. After your normalization step
    # (i.e. after rho0 = Z_sum * rho0 / rho0_sum)
    electron_count_norm = np.sum(rho0) * h ** 3
    print(f"▶ after normalization integrates to {electron_count_norm:.6f} e⁻ (should be {Z_sum})")

    # ==========================================
    # 3. CALCULATE HARTREE POTENTIAL FROM ML DENSITY
    # ==========================================

    print("Calculating Hartree potential from ML density...")

    # 4. Check extremes & noise
    print(f"▶ ML density min/max before clipping: {rho0.min():.3e}, {rho0.max():.3e}")

    # Calculate Hartree potential using the same method as in SCF loop
    # For initial guess, we calculate the Hartree potential of the ML density
    hrhs = (4 * np.pi / h ** 3) * rho0
    # Solve Poisson equation for Hartree potential
    if CG_prec and PRE is not None:
        print("Using preconditioned CG for Hartree potential")
        hpot0, _ = pcg(A, hrhs, np.zeros(n), 200, 1e-4, PRE, 'precLU')
    else:
        print("Using standard CG for Hartree potential")
        hpot0, _ = pcg(A, hrhs, np.zeros(n), 200, 1e-4)

    print(f"Hartree potential statistics: min={np.min(hpot0):.6f}, max={np.max(hpot0):.6f}")

    print("pseudoDiag_ML4Den completed successfully")

    return rho0, hpot0, pot