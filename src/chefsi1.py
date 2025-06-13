import numpy as np
from scipy.linalg import qr, eigh
from Rayleighritz import Rayleighritz
from lanczosForChefsi1 import lanczosForChefsi1
from ch_filter import ch_filter

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def chefsi1(Vin, ritzv, deg, nev, H):
    """
    Python translation of the MATLAB `chefsi1` function.

    Parameters:
    Vin (ndarray): Initial basis vector matrix.
    ritzv (ndarray): Array containing previous Ritz values.
    deg (int): Polynomial degree.
    nev (int): Number of occupied states.
    H (ndarray): Hamiltonian matrix.

    Returns:
    W (ndarray): Filtered eigenvector matrix.
    ritzv (ndarray): Updated Ritz values.
    """
    n, n2 = Vin.shape
    G = np.zeros((n2, n2))
    if nev >= n2:
        raise ValueError('Vin should have more than nev columns')

    # call Lanczos with nev = 6 and m=6 to get the upper interval bound
    upperb = lanczosForChefsi1(H, 6, np.random.randn(n, 1), 6, 0.0)

    # Use max of previous eigenvalues for lower bound
    lowerb = np.max(ritzv)

    # Use the smallest eigenvalue estimate for scaling
    lam1 = np.min(ritzv)

    if lowerb > upperb:
        raise ValueError('Bounds are incorrect')

    # Apply Chebyshev filter to the basis vectors
    W = ch_filter(H, Vin, deg, lam1, lowerb, upperb)

    # Orthonormalize the basis using QR decomposition
    W, R = qr(W, mode='economic')

    # Rayleigh-Ritz projection: compute G = W.T @ H @ W
    Vin = H @ W
    if OPTIMIZATIONLEVEL != 0:
        G = Rayleighritz(Vin, W, n2)
    else:
        for j in range(n2):
            for i in range(j + 1):
                G[i, j] = np.dot(Vin[:, i], W[:, j])
                G[j, i] = G[i, j]
        # TODO: compare performance between c++ and matlab
        # if enableMexFilesTest == 1:
        #     G2 = Rayleighritz(Vin, W, n2)
        #     if (any(abs(G - G2) > 0.000001)):
        #         exception = struct('message', 'Mex file descreptency for Rayleighritz.c', 'identifier', [], 'stack', {mfilename('fullpath') 'filler'});
        #         logError(exception)

    # Diagonalize the Rayleigh-Ritz matrix G
    ritzv, Q = eigh(G)

    # Update the basis W and Ritz values
    W = W @ Q

    return W, ritzv
