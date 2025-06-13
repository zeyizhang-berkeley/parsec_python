import numpy as np
from scipy.linalg import qr, eigh
#from numpy.linalg import eigh
from lanczosForChsubsp import lanczosForChsubsp
from ch_filter import ch_filter
from Rayleighritz import Rayleighritz

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def chsubsp(deg, nev, H):
    """
    Perform the Chebyshev subspace iteration for eigenvalue computation.

    Args:
        deg (int): Polynomial degree.
        nev (int): Number of occupied states.
        H (np.ndarray): Hamiltonian matrix.

    Returns:
        W (np.ndarray): The approximate invariant subspace.
        ritzv (np.ndarray): The new Ritz values.
    """

    # Set initial parameters
    Lanc_steps = max(3 * nev, 450)
    Energ_tol = 0.05
    Max_out_iter = 30
    n = H.shape[0]

    # call Lanczos with Lanc_steps steps for getting the upper interval bound
    W, ritzv, upperb = lanczosForChsubsp(H, nev, np.random.randn(n, 1), Lanc_steps)

    # use previous eigenvalues for lower bound
    tr0 = np.sum(ritzv)

    # Outer iteration loop
    for it in range(Max_out_iter):
        # Use the smallest eigenvalue estimate for scaling
        lam1 = np.min(ritzv)
        lowerb = np.max(ritzv)
        if lowerb > upperb:
            raise ValueError("Bounds are incorrect: lower bound is greater than upper bound")

        # Use polynomial filtering to improve the subspace
        W = ch_filter(H, W, deg, lam1, lowerb, upperb)

        # Rayleigh-ritz projection
        # orthonormalize the basis, should be replaced with better method for real computations
        # Orthonormalize the basis using QR decomposition
        W, R = qr(W, mode='economic')

        # Compute Rayleigh-Ritz projection matrix G = W'*H*W
        Vin = H @ W
        n = Vin.shape[0]
        n2 = Vin.shape[1]

        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, n2)
        else:
            # Manually compute G if optimization is disabled
            G = np.zeros((n2, n2))
            for j in range(n2):
                for i in range(j + 1):
                    G[i, j] = np.dot(Vin[:, i], W[:, j])
                    G[j, i] = G[i, j]  # G is symmetric

            # Optional validation with Mex file comparison
            if enableMexFilesTest == 1:
                G2 = Rayleighritz(Vin, W, n2)
                if np.any(abs(G - G2) > 1e-6):
                    ValueError("Mex file discrepancy for Rayleighritz.c")

        # Compute the eigenvalues and eigenvectors of G
        ritzv, Q = eigh(G)

        # Update the subspace W using the eigenvectors of G
        W = W @ Q

        # Check for convergence based on the sum of eigenvalues
        tr1 = np.sum(ritzv[:nev])
        if np.abs(tr1 - tr0) < Energ_tol * np.abs(tr1):
            break

        tr0 = tr1

    return W, ritzv
