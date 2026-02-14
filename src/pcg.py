import cupy as cp

def pcg(A, rhs, x0, m, tol):
    """
    GPU-accelerated Preconditioned Conjugate Gradient (PCG) solver using CuPy.

    Parameters:
    A   -- CuPy ndarray (n x n)
    rhs -- CuPy ndarray (n x 1)
    x0  -- Initial guess (n x 1)
    m   -- Max iterations
    tol -- Convergence tolerance

    Returns:
    x   -- Solution vector
    its -- Number of iterations
    """
    rhs = rhs.ravel()
    x = x.ravel() 
# .ravel() flatten GPU array to 1D & do plain dot product its faster than cp.dot
#dont do astype its much faster to use a 1d vector
    r = rhs - A @ x
    p = r.copy() #better to just copy to not fw memory instead of z=r, p=z

    ro1 = cp.inner(r, r) #compute inner product, faster than .dot
    tol1 = (tol ** 2) * ro1 

    its = 0
    while its < m and ro1 > tol1:
        its += 1
        Ap = A @ p
        alpha = ro1 / cp.inner(p, Ap) #no dot, inner,
        x += alpha * p
        r -= alpha * Ap
        ro_new = cp.inner(r, r) #no dot, inner
        beta = ro_new / ro1
        p = r + beta * p
        ro1 = ro_new
#dont put cp.cuda.runtime in a solver,
    return x, its