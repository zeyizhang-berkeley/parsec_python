import numpy as np
from scipy import sparse

def spconvert_optimized(D):
    """
    Optimized version of spconvert to convert a matrix D to a sparse format.
    """
    # If already sparse, return as is
    if sparse.issparse(D):
        return D

    # Ensure D is a NumPy array for processing
    if not isinstance(D, np.ndarray):
        raise ValueError("Input D must be a NumPy array or a sparse matrix.")

    na = D.shape[1]

    if na == 3:
        # Construct sparse matrix using columns i, j, re
        rows = D[:, 0].astype(int) - 1  # Adjusting for 0-based indexing
        cols = D[:, 1].astype(int) - 1
        data = D[:, 2]
        S = sparse.coo_matrix((data, (rows, cols)))

    elif na == 4:
        # Construct sparse matrix using columns i, j, re, im
        rows = D[:, 0].astype(int) - 1  # Adjusting for 0-based indexing
        cols = D[:, 1].astype(int) - 1
        data = D[:, 2] + 1j * D[:, 3]
        S = sparse.coo_matrix((data, (rows, cols)))

    else:
        raise ValueError("Wrong array size: Expected 3 or 4 columns in D.")

    return S
