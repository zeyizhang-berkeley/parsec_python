import numpy as np
from splineData import splineData
from preProcess import preProcess
from fspline import fspline
import logging
from fsplevalIO import fsplevalIO

# Set up logging to a file (or console)
logging.basicConfig(level=logging.ERROR, filename='error_log.txt', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MexFileDiscrepancyError(Exception):
    """
    Custom exception class to replicate MATLAB's struct-style error reporting.
    """

    def __init__(self, message, identifier=None, stack=None):
        super().__init__(message)
        self.message = message
        self.identifier = identifier if identifier is not None else []
        self.stack = stack if stack is not None else []


def logError(exception):
    """
    Custom logging function to log errors similarly to MATLAB's logError function.

    Args:
        exception (MexFileDiscrepancyError): Custom exception object with the error details.
    """
    # Log the error details
    logging.error(f"Message: {exception.message}")
    if exception.identifier:
        logging.error(f"Identifier: {exception.identifier}")
    if exception.stack:
        logging.error(f"Stack trace: {exception.stack}")

    # Print the error (optional) - you can replace this with custom logic to handle errors
    print(f"Error logged: {exception.message}")


def pseudoDiag(Domain, Atoms, elem, N_elements):
    """
    Set up initial screening and local ionic pseudopotential.

    Parameters:
    Domain (dict): Contains domain/grid properties.
    Atoms (list): List of atomic species with type and coordinates.
    AtomFuncData (list): List of dictionaries containing atom-specific data.
    elem (DataFrame): Element data from elements_new.csv.
    data_list (list): List of data labels.

    Returns:
    rho0, hpot0, pot (ndarray): Initialized charge density, Hartree potential, and pseudopotential.
    """

    AtomFuncData, data_list = splineData()
    OPTIMIZATIONLEVEL = 0
    enableMexFilesTest = 0

    # Localizing variables
    nx, ny, nz = Domain['nx'], Domain['ny'], Domain['nz']
    h = Domain['h']
    rad = Domain['radius']

    # Pre-allocating memory
    ndim = nx * ny * nz
    pot = np.zeros(ndim)
    rho0 = np.zeros(ndim)
    hpot0 = np.zeros(ndim)

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
        i_charge = data_list.index('charge')
        i_pot_S = data_list.index('pot_S')
        i_hartree = data_list.index('hartree')

        atom_data = AtomFuncData[index]['data']
        x_charg = atom_data[:, 0]
        y_charg = atom_data[:, i_charge]
        x_pot_s = atom_data[:, 0]
        y_pot_s = atom_data[:, i_pot_S]
        x_vhart = atom_data[:, 0]
        y_vhart = atom_data[:, i_hartree]

        # Pre-processing the data
        I = preProcess(y_charg)
        x_charg, y_charg = x_charg[I], y_charg[I]

        I = preProcess(y_pot_s)
        x_pot_s, y_pot_s = x_pot_s[I], y_pot_s[I]

        I = preProcess(y_vhart)
        x_vhart, y_vhart = x_vhart[I], y_vhart[I]

        # Calculating the splines
        z_chg, c_chg, d_chg = fspline(x_charg, y_charg)
        z_p_s, c_p_s, d_p_s = fspline(x_pot_s, y_pot_s)
        z_vht, c_vht, d_vht = fspline(x_vhart, y_vhart)

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
                    rrho = np.zeros(nx)
                    hpot00 = np.zeros(nx)

                    # Initialization of intervals
                    j_ch = 0
                    j_p_s = 0
                    j_vht = 0

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
                            rrho[i_idx], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[i_idx], j_ch)
                            hpot00[i_idx], j_vht = fsplevalIO(z_vht, c_vht, d_vht, x_vhart, r1[i_idx], j_vht)

                            ppot[iPlusOne], j_p_s = fsplevalIO(z_p_s, c_p_s, d_p_s, x_pot_s, r1[iPlusOne], j_p_s)
                            rrho[iPlusOne], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[iPlusOne], j_ch)
                            hpot00[iPlusOne], j_vht = fsplevalIO(z_vht, c_vht, d_vht, x_vhart, r1[iPlusOne], j_vht)
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
                        #         np.any(np.abs(rrho2 - rrho) > 1e-6) or
                        #         np.any(np.abs(hpot002 - hpot00) > 1e-6)):
                        #     # Prepare the stack trace
                        #     current_frame = inspect.currentframe()
                        #     stack_trace = inspect.getouterframes(current_frame)
                        #
                        #     # Raise custom exception with details
                        #     exception = MexFileDiscrepancyError(
                        #         message="Mex file discrepancy for PsuedoDiagLoops",
                        #         identifier=None,  # You can add an identifier if necessary
                        #         stack=[frame.function for frame in stack_trace]  # Extract function names from the stack
                        #     )
                        #     logError(exception)
                        #     raise exception  # Optionally raise the error after logging

                    rrho = np.maximum(0, rrho * (h ** 3) / (4 * np.pi))
                    indx_end = indx + nx

                    pot[indx:indx_end] += ppot
                    rho0[indx:indx_end] += rrho
                    hpot0[indx:indx_end] += hpot00
                    indx = indx_end

    # Normalize rho0
    rho0_sum = np.sum(rho0)
    rho0 = Z_sum * rho0 / rho0_sum

    return rho0, hpot0, pot