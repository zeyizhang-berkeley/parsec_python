from numba import njit
import numpy as np

@njit
def trid_optimized(a, d, c, r):
    """
    Optimized solution to the tridiagonal system A * x = r using:
    1. Numba for JIT compilation
    2. Separate arrays to avoid in-place modifications
    3. Precomputed values to reduce redundant operations
    """
    n = len(d)

    # Preallocate new arrays to avoid modifying inputs
    d_new = d.copy()
    r_new = r.copy()

    # Forward elimination
    for k in range(1, n):
        inv_d = 1.0 / d_new[k - 1]  # Precompute division
        piv = a[k] * inv_d         # Precompute pivot
        d_new[k] -= piv * c[k - 1]
        r_new[k] -= piv * r_new[k - 1]

    # Back substitution
    x = np.zeros_like(r_new)       # Allocate output array
    x[n - 1] = r_new[n - 1] / d_new[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = (r_new[j] - c[j] * x[j + 1]) / d_new[j]

    return x
