import numpy as np
from scipy.linalg import qr, eigh
from Rayleighritz import Rayleighritz

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def lanczosForChsubsp(B, nev, v, m):
    """
    Perform the Lanczos algorithm for eigenvalue computation.

    Args:
        B (np.ndarray): Input matrix.
        nev (int): Number of desired eigenvalues.
        v (np.ndarray): Initial vector.
        m (int): Number of Lanczos steps.

    Returns:
        W (np.ndarray): Matrix of approximate eigenvectors.
        lam (np.ndarray): Approximate eigenvalues.
        bound (float): An upper bound for the eigenvalues.
    """

    reorth = 0
    tol = 1e-5

    # Enable or disable convergence testing based on matrix size
    if B.shape[0] > 35000:
        enableConvergenceTest = 0
    else:
        enableConvergenceTest = 1

    # Matrix dimensions and normalization of the initial vector
    n = B.shape[0]
    v = v / np.linalg.norm(v)
    v1 = v
    v0 = np.zeros(n)
    k = 0
    bet = 0
    ll = 0

    # Pre-allocating arrays
    Tmat = np.zeros((m + 1, m + 1))
    tr = []
    rr = []
    X1 = []
    #indx = []
    bound = 0.0
    VV = np.zeros((n, m))
    if OPTIMIZATIONLEVEL != 0:
        ValueError("not implemented yet!")
        # # Using a custom memory allocation (placeholder for actual implementation)
        # rr, X1, indx, k, bound = lanWhileLoopForChsubsp(m, v0, v1, B, reorth, nev, VV, Tmat)
    else:
        while k < m:
            k += 1
            VV[:, k - 1] = v1

            v = B @ v1 - bet * v0
            # if enableMexFilesTest == 1:
            #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
            #     if np.any(np.abs(v - v2) > 1e-6):
            #         ValueError("Mex file discrepancy for BTimesv1MinusbetTimesv0")

            # Calculate alpha
            alp = np.dot(v1, v)

            if enableMexFilesTest == 0:
                v = v - alp * v1
            # else:
            #     vTemp = v - alp * v1
            #     vTemp2 = vMinusalpTimesv1(v, v1, alp)
            #     v = vTemp
            #     if np.any(np.abs(vTemp - vTemp2) > 1e-6):
            #         ValueError("Mex file discrepancy for vMinusalpTimesv1")

            # Reorthogonalization, if needed
            if reorth:
                subVV = VV[:, :k]
                v = v - subVV @ (subVV.T @ v)

            # Normalize the vector and update variables
            bet = np.linalg.norm(v)
            v0 = v1
            inverseOfbet = 1.0 / bet
            v1 = v * inverseOfbet

            # Update Tmat
            Tmat[k - 1, k - 1] = alp
            Tmat[k, k - 1] = bet
            Tmat[k - 1, k] = bet

            NTest = min(8 * nev, m)

            if enableConvergenceTest and (((k >= NTest) and (k % 10 == 0)) or k == m):
                if k != m:
                    rr = eigh(Tmat[:k, :k])
                else:
                    rr, X1 = eigh(Tmat[:k, :k])

                bound = np.max(np.abs(rr)) + bet
                tr1 = np.sum(rr[:nev])
                ll += 1
                tr[ll - 1] = tr1

            # Convergence criterion
            if enableConvergenceTest and (ll > 1 and (np.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2])):
                rr, X1 = eigh(Tmat[:k, :k])
                break

        if enableConvergenceTest == 0:
            rr, X1 = eigh(Tmat[:k, :k])

    del Tmat
    del v
    del v0
    del v1

    # Save eigenvalues and compute eigenvectors
    if reorth:
        lam = rr[:nev]
        Y = X1[:, :nev]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
    else:
        rrCopy = rr.copy()

        # Find the number of unique eigenvalues
        counter = 0
        # TODO: check corresponding matlab code, maybe wrong?
        while counter < len(rrCopy) - 1:
            rrCopy = np.delete(rrCopy, np.where(np.abs(rrCopy[counter] - rrCopy[counter+1:]) < 1e-5)[0] + counter + 1)
            counter += 1

        mevMultiplier = min(1.8 ** (len(rr) / len(rrCopy)), 3.2)
        mev = int(min(nev * mevMultiplier, k))

        Y = X1[:, :mev]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
        del VV  # Clean up memory

        # Orthogonalize W using QR decomposition
        W, G, _ = qr(W, mode='economic')

        # Calculate the Rayleigh-Ritz matrix G
        Vin = B @ W
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, mev)
        else:
            G = np.zeros((mev, mev))
            for j in range(mev):
                for i in range(j + 1):
                    G[i, j] = np.dot(Vin[:, i], W[:, j])
                    G[j, i] = G[i, j]

            if enableMexFilesTest == 1:
                G2 = Rayleighritz(Vin, W, mev)
                if np.any(abs(G - G2) > 1e-6):
                    ValueError("Mex file discrepancy for Rayleighritz.c")

        rr, X1 = eigh(G)
        lam = rr[:nev]
        Y = X1[:, :nev]
        W = W @ Y

    return W, lam, bound

