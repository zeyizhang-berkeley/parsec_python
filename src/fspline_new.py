from numba import njit
import numpy as np
from trid_new import trid_optimized

@njit
def fspline_optimized(xi, yi):
    """
    Optimized cubic spline coefficient calculation using Numba.
    """
    xi = xi.flatten()  # Ensure xi is a 1D array
    yi = yi.flatten()  # Ensure yi is a 1D array
    np1 = len(xi)      # Number of points
    n = np1 - 1        # Number of intervals

    # Get the step sizes (h)
    h = xi[1:np1] - xi[0:n]

    # Calculate the right-hand-side (r)
    r = (yi[1:np1] - yi[0:n]) / h  # First derivative differences
    r = r[1:n] - r[0:n - 1]        # Second derivative differences

    # Set up the tridiagonal matrix system
    L = h[0:n - 1]                # Lower diagonal
    U = h[1:n]                    # Upper diagonal
    D = 2 * (L + U)               # Main diagonal

    # Solve the tridiagonal system using a custom Numba-compatible solver
    z = trid_optimized(L, D, U, r)  # Use the Numba-optimized trid function

    # Set the end values to 0
    z = np.concatenate((np.array([0]), z, np.array([0])))

    # Calculate the cubic spline coefficients
    c = (yi[0:n] / h) - (z[0:n] * h / 6)
    d = (yi[1:np1] / h) - (z[1:np1] * h / 6)

    return z, c, d
