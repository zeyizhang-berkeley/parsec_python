import numpy as np
from numba import njit

@njit
def preProcess_numba(v):
    """
    Optimized pre-process function using Numba for speed.
    """
    eps = 10e-6
    I = [0]  # Start with the first index

    # Iterate over the vector and check differences
    for i in range(1, len(v) - 1):
        if np.abs(v[i] - v[i - 1]) > eps:
            I.append(i)

    # Always keep the last index
    I.append(len(v) - 1)

    return np.array(I)
