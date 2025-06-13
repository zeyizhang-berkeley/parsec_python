from numba import njit

@njit
def fsplevalIO_optimized(z, c, d, xi, x, j_in):
    """
    Optimized evaluation of free spline function at x using Numba.

    Parameters:
    z (ndarray): Spline z coefficients.
    c (ndarray): Spline c coefficients.
    d (ndarray): Spline d coefficients.
    xi (ndarray): Spline xi points.
    x (float): Point at which to evaluate the spline.
    j_in (int): Interval index to try first.

    Returns:
    y (float): Evaluated value of the spline at x.
    j_out (int): Output interval index for the next call.
    """
    n = len(xi)

    # Clamp x to be within the range of xi
    if x < xi[0]:
        x = xi[0]
        j_in = 0
    elif x > xi[-1]:
        x = xi[-1]
        j_in = n - 2

    # Ensure j_in is valid
    if j_in < 0 or j_in >= n - 1:
        j_in = 0

    j_out = j_in

    # Perform binary search if x is not in the interval [xi[j_in], xi[j_in + 1]]
    if not (xi[j_in] <= x <= xi[j_in + 1]):
        ind_low = 0
        ind_high = n - 1
        while ind_high - ind_low > 1:
            ind_middle = (ind_high + ind_low) // 2
            if x < xi[ind_middle]:
                ind_high = ind_middle
            else:
                ind_low = ind_middle
        j_out = ind_low

    # Compute the spline value
    t1 = xi[j_out + 1] - x
    t2 = x - xi[j_out]
    h_j_out = xi[j_out + 1] - xi[j_out]

    y = (t1 * (z[j_out] * t1 * t1 / h_j_out + c[j_out]) +
         t2 * (z[j_out + 1] * t2 * t2 / h_j_out + d[j_out]))

    return y, j_out + 1
