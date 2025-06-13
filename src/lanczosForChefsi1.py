import numpy as np
from scipy.linalg import eigh

enableMexFilesTest = 0
def lanczosForChefsi1(B, nev, v, m, tol=1e-6, reorth=False):
    """
    Python translation of the MATLAB `lanczosForChefsi1` function.
    Estimates the upper bound of eigenvalues using Lanczos iterations.

    Parameters:
    B (ndarray): The matrix to perform Lanczos on.
    nev (int): Number of desired eigenvalues.
    v (ndarray): Initial vector.
    m (int): Number of Lanczos steps.
    tol (float): Tolerance for convergence.
    reorth (bool): Whether to perform full reorthogonalization.

    Returns:
    float: Estimated upper bound for the eigenvalues.
    """
    bound = 0

    n = B.shape[0]
    v = v / np.linalg.norm(v)
    #print("v:", v.shape)
    v1 = v
    v0 = np.zeros((n,1))
    k = 0
    bet = 0
    ll = 0

    # Pre-allocate memory
    VV = []
    if reorth:
        VV = np.zeros((n, m))

    Tmat = np.zeros((m + 1, m + 1))
    tr = np.zeros(m)

    # Main Lanczos loop
    while k < m:
        k += 1

        if reorth:
            VV[:, k - 1] = v1

        # Lanczos iteration
        #print("B shape: ", B.shape)
        #print("v1: ", v1.shape)
        #print("v0: ", v0.shape)
        #print("bet", bet)
        v = B @ v1 - bet * v0
        #print("v: ", v.shape)

        # TODO: compare performance between c++ and matlab
        # if enableMexFilesTest == 1:
        #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
        #     if (any(abs(v - v2) > 0.000001))
        #         exception = struct('message', 'Mex file descreptency for BTimesv1MinusbetTimesv0', 'identifier', [],
        #                            'stack', {mfilename('fullpath')
        #         'filler'})
        #         logError(exception)

        alp = (np.dot(v1.T, v).item()).real
        #print("alp: ", alp)

        if enableMexFilesTest == 0:
            v -= alp * v1  # Equivalent to: v = v - alp * v1
        # else:
        # TODO: compare performance between c++ and matlab

        #     vTemp = v - alp * v1
        #     vTemp2 = vMinusalpTimesv1(v, v1, alp)
        #     v = vTemp
        #     if (any(abs(vTemp - vTemp2) > 0.000001))
        #         exception = struct('message', 'Mex file descreptency for vMinusalpTimesv1', 'identifier', [], 'stack',
        #                            {mfilename('fullpath')
        #         'filler'})
        #         logError(exception)

        # Reorthogonalization if required
        if reorth:
            t = VV[:, :k].T @ v
            v -= VV[:, :k] @ t

        # Normalize and store new vector
        bet = np.linalg.norm(v)
        v0 = v1
        v1 = v / bet

        # Update the tridiagonal matrix
        Tmat[k - 1, k - 1] = alp
        Tmat[k, k - 1] = bet
        Tmat[k - 1, k] = bet

        # Convergence testing
        NTest = min(5 * nev, m)
        if (k >= NTest and k % 10 == 0) or k == m:
            rr, _ = eigh(Tmat[:k, :k])  # Calculate eigenvalues of Tmat

            # upper bound used by chef_si -- not used otherwise.
            bound = max(np.abs(rr)) + bet
            tr1 = np.sum(rr[:nev])
            ll += 1
            tr[ll - 1] = tr1

            # Check convergence based on sum of eigenvalues
            if ll > 1 and np.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2]:
                break

    return bound
