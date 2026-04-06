import cupy as cp
from scipy.linalg import eigh


def lancz_uppbnd(n, A, k=6):
    k = min(max(k, 6), 10)

    T = cp.zeros((k, k), dtype=cp.float32)
    v = cp.random.rand(n, 1, dtype=cp.float32)
    v = v / cp.linalg.norm(v)

    tol = 2.5e-16
    upperb = cp.zeros((3, k), dtype=cp.float32)

    f = A @ v
    alpha = cp.vdot(v, f)
    f = f - alpha * v
    T[0, 0] = alpha
    beta = cp.linalg.norm(f)

    upperb[0, 0] = alpha + beta
    upperb[1, 0] = upperb[0, 0]
    upperb[2, 0] = upperb[0, 0]

    isbreak = 0
    ritzv = None
    X = None

    for j in range(2, k + 1):
        if beta > tol:
            v0 = v
            v = f / beta
            f = A @ v
            f = f - beta * v0
            alpha = cp.vdot(v, f)
            f = f - alpha * v
            T[j - 1, j - 2] = beta
            T[j - 2, j - 1] = beta
            T[j - 1, j - 1] = alpha
        else:
            isbreak = 1
            print(f"j = {j}, invariant subspace found")
            break

        beta = cp.linalg.norm(f)
        if isbreak != 1:
            ritzv_host, X_host = eigh(cp.asnumpy(T[0 : j + 1, 0 : j + 1]))
        else:
            ritzv_host, X_host = eigh(cp.asnumpy(T[0:j, 0:j]))
        ritzv = cp.asarray(ritzv_host, dtype=cp.float32)
        X = cp.asarray(X_host, dtype=cp.float32)

        if beta < 1e-2:
            beta = beta * 10

        upperb[0, j - 1] = ritzv[-1] + beta
        upperb[1, j - 1] = ritzv[-1] + cp.abs(X[-1, -1]) * beta
        upperb[2, j - 1] = ritzv[-1] + cp.max(cp.abs(X[-1, :])) * beta

    uppbnd = (upperb[0, j - 1] + upperb[1, j - 1]) / 2
    uppbnd = (upperb[2, j - 1] + uppbnd) / 2

    return uppbnd, ritzv
