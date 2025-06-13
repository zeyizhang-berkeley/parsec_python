import numpy as np

def FermiDirac(lam, EF, Temp, Nelec):
    """
    Python translation of the MATLAB `FermiDirac` function.
    Evaluates the Fermi-Dirac distribution for the given eigenvalues.

    Parameters:
    lam (ndarray): Array of eigenvalues.
    EF (float): Fermi level.
    Temp (float): Temperature in energy units.
    Nelec (float): Total number of electrons.

    Returns:
    fe (float): Error related to the sum of occupation numbers and Nelec.
    occup (ndarray): Occupation numbers for each eigenvalue.
    """
    # Constants
    kT = Temp * 6.33327186e-06  # Scaling temperature
    spin = 1  # Spin factor; adjust as needed

    # Calculate the occupation numbers using the Fermi-Dirac formula
    t = 1 + np.exp((lam - EF) / kT)
    occup = spin / t

    # Compute the deviation from the expected number of electrons
    fe = np.sum(occup) - Nelec / 2

    return fe, occup
