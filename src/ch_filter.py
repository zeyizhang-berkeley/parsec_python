
OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def ch_filter(A, x, deg, lam1, low, high, optimization_level=0, enable_mex_files_test=False):
    """
    Python translation of the MATLAB `ch_filter` function.
    Applies the Chebyshev filter to a given vector `x` using matrix `A`.

    Parameters:
    A (ndarray): The matrix to filter with.
    x (ndarray): The vector to be filtered.
    deg (int): Degree of the polynomial.
    lam1 (float): Estimate of the lowest eigenvalue.
    low (float): Lower bound of the interval.
    high (float): Upper bound of the interval.
    optimization_level (int): Global variable `OPTIMIZATIONLEVEL`.
    enable_mex_files_test (bool): Global variable `enableMexFilesTest`.

    Returns:
    ndarray: The filtered vector.
    """

    # Calculate scaling parameters for Chebyshev polynomials
    e = (high - low) / 2
    c = (high + low) / 2
    sigma1 = e / (lam1 - c)
    sigma = sigma1

    # TODO: compare performance between c++ and matlab and implement C++ version
    # If MEX file testing is enabled, log discrepancies
    # if enableMexFilesTest==1:
    #     # Degree 1 Chebyshev term
    #     y = (A @ x - c * x) * (sigma1 / e)
    #     y2 = chebyshevfilterDegree1(A, x, c, sigma1, e)
    #     if not np.allclose(y, y2, atol=1e-6):
    #         raise ValueError("Mex file discrepancy for chebyshevfilterDegree1")
    #
    #     twoDividedbysigma1 = 2 / sigma1
    #     inverseOfe = 1 / e
    #
    # # Loop for higher degree terms
    #     for i in range(2, deg + 1):
    #         # Calculate the next Chebyshev polynomial coefficients
    #         sigma_new = 1 / (twoDividedbysigma1 - sigma)
    #         t1 = 2 * sigma_new * inverseOfe
    #         t2 = sigma * sigma_new
    #
    #         # Compute the new filtered vector
    #         ynew = (A @ y - c * y) * t1 - t2 * x
    #         ynew2 = chebyshevfilterDegreeN(A, x, y, c, t1, t2)
    #
    #         if not np.allclose(ynew, ynew2, atol=1e-6):
    #             raise ValueError("Mex file discrepancy for chebyshevfilterDegreeN")
    #
    #         # Update the vectors for the next iteration
    #         x = y
    #         y = ynew
    #         sigma = sigma_new
    # else:
    #     if OPTIMIZATIONLEVEL !=0:
    #         y = chebyshevfilterDegree1(A, x, c, sigma1, e)
    #
    #         twoDividedbysigma1 = 2 / sigma1
    #         inverseOfe = 1 / e
    #         for i in range(2, deg + 1):
    #             sigma_new = 1 / (twoDividedbysigma1 - sigma)
    #             t1 = 2 * sigma_new * inverseOfe
    #             t2 = sigma * sigma_new
    #             ynew = chebyshevfilterDegreeN(A, x, y, c, t1, t2)
    #
    #             x = y
    #             y = ynew
    #             sigma = sigma_new
    #     else:
    y = (A @ x - c * x) * (sigma1 / e)

    twoDividedbysigma1 = 2 / sigma1
    inverseOfe = 1 / e

    for i in range(2, deg + 1):
        sigma_new = 1 / (twoDividedbysigma1 - sigma)
        t1 = 2 * sigma_new * inverseOfe
        t2 = sigma * sigma_new
        ynew = (A @ y - c * y) * t1 - t2 * x

        x = y
        y = ynew
        sigma = sigma_new
    return y
