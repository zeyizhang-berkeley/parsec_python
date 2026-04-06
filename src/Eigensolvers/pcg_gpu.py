import cupy as cp
from cupy._core import set_reduction_accelerators, set_routine_accelerators
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as sps


set_reduction_accelerators([])
set_routine_accelerators([])


def _to_gpu_matrix(A):
    if isinstance(A, cp.ndarray):
        return A.astype(cp.float32, copy=False)
    if isinstance(A, cpsparse.spmatrix):
        return A.astype(cp.float32)
    if sps.issparse(A):
        return cpsparse.csr_matrix(A.astype("float32"))
    return cp.asarray(A, dtype=cp.float32)


def _apply_preconditioner(preconditioner, precfun, residual):
    if preconditioner is None or precfun is None:
        return residual

    residual_host = cp.asnumpy(residual)
    if callable(precfun):
        z_host = precfun(preconditioner, residual_host)
    elif isinstance(precfun, str):
        if precfun == "precLU" and hasattr(preconditioner, "solve"):
            z_host = preconditioner.solve(residual_host)
        else:
            raise TypeError(f"Unsupported GPU preconditioner label: {precfun}")
    elif hasattr(precfun, "solve"):
        z_host = precfun.solve(residual_host)
    else:
        raise TypeError("GPU preconditioner must be callable or expose a solve() method")

    return cp.asarray(z_host, dtype=cp.float32).ravel()


def pcg(A, rhs, x0, m, tol, *args):
    """
    Solve A * x = rhs on the GPU using the same PCG iteration as the CPU path.
    """
    if len(args) == 2:
        PRE, precfun = args
    else:
        PRE = precfun = None

    A = _to_gpu_matrix(A)
    rhs = cp.asarray(rhs, dtype=cp.float32).ravel()
    x = cp.asarray(x0, dtype=cp.float32).ravel()

    r = rhs - A @ x
    z = _apply_preconditioner(PRE, precfun, r)
    p = z.copy()
    ro1 = cp.inner(z, r)
    tol1 = (tol**2) * ro1

    its = 0
    while its < m and bool((ro1 > tol1).item()):
        its += 1
        ro = ro1
        ap = A @ p
        alp = ro / cp.inner(ap, p)

        x += alp * p
        r -= alp * ap

        z = _apply_preconditioner(PRE, precfun, r)
        ro1 = cp.inner(z, r)
        bet = ro1 / ro
        p = z + bet * p

    return x, its
