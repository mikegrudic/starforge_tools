"""Approximations for useful special mathematical functions"""
from numba import njit, vectorize, float32, float64
import numpy as np
from scipy.special import roots_legendre


@vectorize(fastmath=True)
def logpoisson(counts, expected_counts):
    """Fast computation of log of poisson PMF, using Ramanujan's Stirling-type
    approximation to the factorial"""
    counts = int(counts + 0.5)  # round counts
    if expected_counts == 0:
        if counts == 0:
            return 0
        else:
            return -np.inf
    if counts == 0:
        logfact = 0
    elif counts < 10:
        fact = counts
        for i in range(2, counts):  # factorial
            fact *= i
        logfact = np.log(fact)
    else:  # stirling-type approximation due to Ramanujan
        # cast counts to avoid integer overflow inside 3rd order term
        counts = float(counts)
        logfact = (
            np.log(counts) * counts
            - counts
            + np.log(counts * (1 + 4 * counts * (1 + 2 * counts))) / 6
            + np.log(np.pi) * 0.5
        )
    return counts * np.log(expected_counts) - expected_counts - logfact


@njit(fastmath=True, error_model="numpy")
def planck3_upper_series(x):
    """Series solution for upper Planck integral norm * x^3/(exp(x)-1) from x to
    infinity evaluated to machine precision. Most efficient for large x.
    """
    tol = 2.220446049250313e-16
    series_sum = 0
    err = 1e100
    expx_inv = np.exp(-x)
    expx_inv_power = expx_inv  # will cumulatively multiply at each iteration
    n = 1
    while np.abs(err) > tol * series_sum:
        nx = n * x
        err = (6 + nx * (6 + nx * (3 + nx))) * expx_inv_power / (n * n * n * n)
        expx_inv_power *= expx_inv
        series_sum += err
        n += 1

    return series_sum * 0.15398973382026503


@njit(fastmath=True, error_model="numpy")
def planck2_upper_series(x):
    """Series solution for integral of (1/norm) x^2/(exp(x)-1) x to infinity
    evaluated to machine precision. Most efficient for large x.
    """
    tol = 2.220446049250313e-16
    series_sum = 0
    err = 1e100
    expx_inv = np.exp(-x)
    expx_inv_power = expx_inv  # will cumulatively multiply at each iteration
    n = 1
    while np.abs(err) > tol * series_sum:
        nx = n * x
        err = (2 + nx * (2 + nx)) * expx_inv_power / (n * n * n)
        expx_inv_power *= expx_inv
        series_sum += err
        n += 1

    return series_sum * 0.15398973382026503


@njit(fastmath=True, error_model="numpy")
def planck3_lower_series(x):
    """Series solution for Planck function integral from 0 to x,
    most efficient for small x, valid to machine precision on [0,3]
    """
    coeffs = np.array(
        [
            0.3333333333333333,
            -0.125,
            0.016666666666666666,
            -0.0001984126984126984,
            3.6743092298647855e-6,
            -7.515632515632516e-8,
            1.6059043836821615e-9,
            -3.522793425791662e-11,
            7.872080312167458e-13,
            -1.784042261222412e-14,
            4.088600979179926e-16,
            -9.455950863295921e-18,
            2.203601131344092e-19,
            -5.168320254004638e-21,
            1.2188644964239545e-22,
            -2.888231428076628e-24,
            6.87258318890207e-26,
            -1.641368762534915e-27,
            3.932898582742878e-29,
            -9.451269078629001e-31,
            2.2772522578280595e-32,
            -5.500052129536349e-34,
            1.3312603916626966e-35,
            -3.22862741376232e-37,
            7.844404337661609e-39,
            -1.909088837773861e-40,
        ]
    )

    x_sqr = x * x
    val = coeffs[-1] * x_sqr

    n = len(coeffs) - 1
    while n >= 3:
        val = x_sqr * (val + coeffs[n])
        n -= 1
    while n >= 0:
        val = x * (val + coeffs[n])
        n -= 1
    val *= x_sqr

    return val * 0.15398973382026503


@njit(fastmath=True, error_model="numpy")
def planck2_lower_series(x):
    """Series solution for integral norm * x^2/(exp(x)-1) from 0 to x,
    most efficient for large x, valid to machine precision on [0,3]

    Normalization is 1/integral(x^3 exp(x) - 1) = 15/pi^4, such that
    this can be used to compute average photon energies or photon
    numbers
    """
    coeffs = np.array(
        [
            0.5,
            -0.166666666666667,
            0.0208333333333333,
            -0.000231481481481481,
            4.13359788359788e-6,
            -8.26719576719577e-8,
            1.73972974898901e-9,
            -3.77442152763392e-11,
            8.36408533167792e-13,
            -1.88315572017921e-14,
            4.29303102813892e-16,
            -9.88576681162755e-18,
            2.2954178451501e-19,
            -5.36710180223559e-21,
            1.26239537129624e-22,
            -2.98450580901252e-24,
            7.08735141355526e-26,
            -1.68964431437418e-27,
            4.04214576559685e-29,
            -9.69998668596134e-31,
            2.33418356427376e-32,
        ]
    )

    x_sqr = x * x
    val = coeffs[-1] * x_sqr

    n = len(coeffs) - 1
    while n >= 3:
        val = x_sqr * (val + coeffs[n])
        n -= 1
    while n >= 0:
        val = x * (val + coeffs[n])
        n -= 1
    val *= x

    return val * 0.15398973382026503


@vectorize(
    [float32(float32, float32), float64(float64, float64)],
    target="parallel",
    fastmath=True,
)
def planck3_integral(x1, x2):
    """Returns the definite integral of the normalized (frequency)
    Planck function norm * x^3/(exp(x)-1) from x1 to x2, accurate to machine
    precision
    """
    if x2 < x1:
        # assume x1 < x2 throughout, and just remember the original parity
        x1, x2 = x2, x1
        sign = -1
    else:
        sign = 1

    cutoff_upper_lower = 3  # where to switch between small x and large x series approx

    # initialize flag for if we need to take care with precision
    upper_precision = False

    if x1 == 0:
        f1 = 0
    else:
        if x1 < cutoff_upper_lower:
            f1 = planck3_lower_series(x1)
        else:
            upper_precision = True
            f1 = planck3_upper_series(x1)

    if x2 == np.inf:
        if upper_precision:
            return f1 * sign
        return (1 - f1) * sign
    if x2 < cutoff_upper_lower:
        f2 = planck3_lower_series(x2)
        return (f2 - f1) * sign
    f2 = planck3_upper_series(x2)
    if upper_precision:
        return (f1 - f2) * sign
    return (1 - f1 - f2) * sign


@vectorize(
    [float32(float32, float32), float64(float64, float64)],
    target="parallel",
    fastmath=True,
)
def planck2_integral(x1, x2):
    """Returns the definite integral of the normalized (frequency)
    Planck function norm * x^2/(exp(x)-1) from x1 to x2, accurate to machine
    precision

    Normalization is 1/integral(x^3 exp(x) - 1) = 15/pi^4, such that
    this can be used to compute average photon energies or photon
    numbers
    """
    if x2 < x1:
        # assume x1 < x2 throughout, and just remember the original parity
        x1, x2 = x2, x1
        sign = -1
    else:
        sign = 1

    cutoff_upper_lower = 3  # where to switch between small x and large x series approx

    # initialize flag for if we need to take care with precision
    upper_precision = False

    if x1 == 0:
        f1 = 0
    else:
        if x1 < cutoff_upper_lower:
            f1 = planck2_lower_series(x1)
        else:
            upper_precision = True
            f1 = planck2_upper_series(x1)

    if x2 == np.inf:
        if upper_precision:
            return f1 * sign
        return (1 - f1) * sign
    if x2 < cutoff_upper_lower:
        f2 = planck2_lower_series(x2)
        return (f2 - f1) * sign
    f2 = planck2_upper_series(x2)
    if upper_precision:
        return (f1 - f2) * sign
    return (1 - f1 - f2) * sign


@njit(fastmath=True, error_model="numpy")
def planck_function3(x):
    """Planck function normalized to integrate to 1"""
    return x**3 / (np.exp(x) - 1) * 0.15398973382026507


roots, weights, mu = roots_legendre(12, mu=True)


def planck_integral_gaussian_quadrature(a, b):
    """Integrates the Planck function from a to b using 9th-order Gaussian quadrature"""
    mu = 2
    roots_mapped = ((b - a) / 2)[:, None] * (roots + 1) + a[:, None]
    return np.sum(planck_function3(roots_mapped) * weights, axis=1) * (b - a) / mu
