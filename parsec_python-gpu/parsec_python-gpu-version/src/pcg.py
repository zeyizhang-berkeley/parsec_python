import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import scipy.sparse as sps


def _to_gpu_matrix(A):
    if isinstance(A, cp.ndarray):
        return A.astype(cp.float32)
    if isinstance(A, cpsparse.spmatrix):
        return A.astype(cp.float32)
    if sps.issparse(A):
        return cpsparse.csr_matrix(A.astype(np.float32))
    return cp.asarray(A, dtype=cp.float32)


def pcg(A, rhs, x0, m, tol):
    """
    GPU-accelerated conjugate gradient solver that accepts SciPy/CuPy inputs
    and returns a NumPy vector for the surrounding SCF code.
    """
    A = _to_gpu_matrix(A)
    rhs = cp.asarray(rhs, dtype=cp.float32).ravel()
    x = cp.asarray(x0, dtype=cp.float32).ravel()

    r = rhs - A @ x
    p = r.copy()

    ro1 = cp.inner(r, r)
    tol1 = (tol ** 2) * ro1

    its = 0
    while its < m and ro1 > tol1:
        its += 1
        Ap = A @ p
        alpha = ro1 / cp.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        ro_new = cp.inner(r, r)
        beta = ro_new / ro1
        p = r + beta * p
        ro1 = ro_new

    return cp.asnumpy(x), its
