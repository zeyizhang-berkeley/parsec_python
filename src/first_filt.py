import numpy as np
from scipy.linalg import qr, eigh
#from numpy.linalg import eigh
from lancz_uppbnd import lancz_uppbnd
from Rayleighritz import Rayleighritz

def first_filt(nev, H, polm=10):
    """
    Python translation of the MATLAB first_filt function.

    Parameters:
    nev   -- Number of Ritz values to compute (>= # of occupied states).
    H     -- The Hamiltonian matrix.
    polm  -- Polynomial degree (default is 10).

    Returns:
    ritzv -- Contains the new Ritz values.
    W     -- The orthonormal basis of an approximate subspace.
    """
    global tr0
    OPTIMIZATIONLEVEL=0

    DEBUG = False
    Energ_tol = 0.08
    if polm < 10 or polm > 15:
        polm = 10

    max_iter = max(min(int(60 / polm), 5), 3)
    n = H.shape[0]

    # Call upper bound estimator to compute an upper bound
    upperb, ritzv = lancz_uppbnd(n, H)
    lowerb = 0.8 * ritzv[0] + 0.2 * ritzv[-2]

    if DEBUG:
        print(f'polm={polm}, nev={nev}, sizeH={n}, max_iter={max_iter}')
        print(f'upperb={upperb},  lowerb={lowerb}')

    # Filter random vectors
    W = np.random.randn(n, nev)

    for it in range(max_iter):
        W = cheb_filter_slim(H, W, polm, lowerb, upperb)

        # QR Decomposition
        # TODO: check parameters
        W, G = qr(W, mode='economic')
        Vin = H @ W
        n2 = Vin.shape[1]
        if n2 != nev:
            raise ValueError('Wrong number of eigenvalues')

        if OPTIMIZATIONLEVEL != 0:
            # TODO: switch to C++ code to compare the performance?
            G = Rayleighritz(Vin, W, n2)
        else:
            # Calculate G using matrix multiplications
            G = np.zeros((n2, n2))
            for j in range(n2):
                for i in range(j + 1):
                    G[i, j] = np.dot(Vin[:, i], W[:, j])
                    G[j, i] = G[i, j]

        # Eigendecomposition of G
        ritzv, Q = eigh(G)
        lowerb = ritzv[-1]
        W = W @ Q

        if it == 0:
            tr0 = np.sum(ritzv[:nev])
        else:
            tr1 = np.sum(ritzv[:nev])
            if abs(tr1 - tr0) <= Energ_tol * abs(tr0):
                break

            if DEBUG and it > 1:
                print(f'iter={it + 1}, lowerb = {lowerb}, ritz_diff={abs(tr1 - tr0)}')

            tr0 = tr1

    return W, ritzv


def cheb_filter_slim(H, x, polm, low, high):
    """
    Chebyshev filtering slim version, non-scaling normalized.

    Parameters:
    H     -- The Hamiltonian matrix.
    x     -- Input matrix.
    polm  -- Polynomial degree.
    low   -- Lower bound.
    high  -- Upper bound.

    Returns:
    y     -- The filtered matrix.
    """
    e = (high - low) / 2
    center = (high + low) / 2

    y = H @ x
    y = (-y + center * x) / e

    for i in range(2, polm + 1):
        ynew = H @ y
        ynew = (-ynew + center * y) * 2 / e - x
        x = y
        y = ynew

    return y
