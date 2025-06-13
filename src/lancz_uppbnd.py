import numpy as np
from scipy.linalg import eigh
#from numpy.linalg import eigh


def lancz_uppbnd(n, A, k=6):
    """
    Python translation of the MATLAB `lancz_uppbnd` function.

    Parameters:
    n  -- Dimension of the matrix.
    A  -- The matrix or a callable that performs matrix-vector multiplication.
    k  -- Number of Lanczos steps to perform (default is 6).

    Returns:
    uppbnd  -- Estimated upper bound for the eigenvalues.
    ritzv   -- Ritz values obtained during the Lanczos iterations.
    """
    k = min(max(k, 6), 10)  # Limit k to be between 6 and 10

    # Initialize the tridiagonal matrix T
    T = np.zeros((k, k))
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    tol = 2.5e-16  # Convergence tolerance
    upperb = np.zeros((3, k))  # Save bounds for each step

    f = A @ v  # Matrix-vector multiplication
    alpha = np.dot(v, f)
    f = f - alpha * v
    T[0, 0] = alpha
    beta = np.linalg.norm(f)

    # Compute bounds for j=1 using Rayleigh quotient
    upperb[:, 0] = alpha + beta

    isbreak = 0

    j = 1
    ritzv = []
    for j in range(1, k):  # Lanczos iteration starts from 2 (index 1)
        if beta > tol:
            v0 = v
            v = f / beta
            f = A @ v
            f = f - beta * v0
            alpha = np.dot(v, f)
            f = f - alpha * v
            T[j, j - 1] = beta
            T[j - 1, j] = beta
            T[j, j] = alpha
        else:
            isbreak = 1
            print(f'j = {j}, invariant subspace found')
            break

        beta = np.linalg.norm(f)

        # Compute eigenvalues of the current tridiagonal matrix
        if (isbreak!=1):
            ritzv, X = eigh(T[:j, :j])
        else:
            ritzv, X = eigh(T[:j-1, :j-1])

        if beta < 1e-2:
            beta *= 10  # Scale beta for better stability

        upperb[0, j] = ritzv[-1] + beta
        upperb[1, j] = ritzv[-1] + abs(X[-1, -1]) * beta
        upperb[2, j] = ritzv[-1] + max(abs(X[-1, :])) * beta

    uppbnd = (upperb[0, j] + upperb[1, j]) / 2
    uppbnd = (upperb[2, j] + uppbnd) / 2

    return uppbnd, ritzv
