"""Approximations for useful special mathematical functions"""

from numba import njit, vectorize, float32, float64, int32, int64, boolean
import numpy as np
from astropy.constants import h, c, k_B
from astropy import units as u


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


# pre-computed lookup table of complete Planck integrals; entry i is integral of order i+1
PLANCK_NORM = np.array(
    [
        1.6449340668482264e00,
        2.4041138063191885e00,
        6.4939394022668289e00,
        2.4886266123440880e01,
        1.2208116743813390e02,
        7.2601147971498449e02,
        5.0605498752376398e03,
        4.0400978398747633e04,
        3.6324091142238257e05,
        3.6305933116066284e06,
        3.9926622987731084e07,
        4.7906037988983148e08,
        6.2274021934109726e09,
        8.7180957830172058e10,
        1.3076943522189138e12,
        2.0922949679481512e13,
        3.5568878585922375e14,
        6.4023859228189210e15,
        1.2164521645363939e17,
        2.4329031685078615e18,
    ]
)


@njit(fastmath=True, error_model="numpy")
def planck_function(x, p=3):
    """General Planck function x^p / (exp(x) - 1)"""
    return x**p / np.expm1(x)


@njit(fastmath=True, error_model="numpy")
def planck_norm(p):
    """Complete Planck integral from 0 to infinity"""
    return PLANCK_NORM[p - 1]


@njit(fastmath=True, error_model="numpy")
def planck_upper_series(x1: float, x2: float = np.inf, p: int = 3):
    """Series solution for Planck integral x^p/(exp(x)-1) from x1 to
    x2 evaluated to machine precision. Most efficient for large x.

    Parameters
    ----------
    x1: float
        Lower integral limit
    x2: float, optional
        Upper integration limit (default: inf)
    p: int, optional
        Exponent of Planck function (default: 3)

    Returns
    -------
    Planck integral evaluated to machine precision
    """
    tol = 2.220446049250313e-16
    series_sum = 0.0
    term = 1e100
    term1 = term2 = 1.0

    if x1 > x2:  # assume x2 is the larger in magnitude
        x1, x2 = x2, x1
        sign = -1
    else:
        sign = 1

    # will cumulatively multiply exp(-x) at each iteration
    expx1_inv = expx1_inv_power = np.exp(-x1)

    if x2 == np.inf:
        expx2_inv = expx2_inv_power = 0.0
    else:
        expx2_inv = expx2_inv_power = np.exp(-x2)

    n = 1

    # now sum the terms. Note this is the general case; we could simplify
    # if we know p a priori and make this a little faster
    while np.abs(term) > tol * np.abs(series_sum):
        if x2 == np.inf:
            nx1 = n * x1
            term1 = 1.0
            m = 1
            nprod = n
            for i in range(p, 0, -1):
                m *= i
                term1 = term1 * nx1 + m
                nprod *= n
            term1 /= nprod
            term1 *= expx1_inv_power
            expx1_inv_power *= expx1_inv
            term2 = 0
        else:  # both x1 and x2 are finite
            nx1 = n * x1
            nx2 = n * x2
            term1 = term2 = 1.0
            m = 1
            nprod = n
            for i in range(p, 0, -1):
                m *= i
                term1 = term1 * nx1 + m
                term2 = term2 * nx2 + m
                nprod *= n
            term1 /= nprod
            term2 /= nprod
            term1 *= expx1_inv_power
            expx1_inv_power *= expx1_inv
            term2 *= expx2_inv_power
            expx2_inv_power *= expx2_inv

        term = term1 - term2
        series_sum += term
        n += 1

    return sign * series_sum


@njit(fastmath=True, error_model="numpy")
def planck_gaussquad(a: float, b: float, p: int):
    """Gaussian quadrature for Planck function integral from a to b."""
    mu = 2.0
    roots = np.array(
        [
            -0.9815606342467192,
            -0.9041172563704749,
            -0.7699026741943047,
            -0.5873179542866175,
            -0.3678314989981801,
            -0.125233408511469,
            0.125233408511469,
            0.3678314989981801,
            0.5873179542866175,
            0.7699026741943047,
            0.9041172563704749,
            0.9815606342467192,
        ]
    )

    weights = np.array(
        [
            0.0471753363865132,
            0.1069393259953178,
            0.1600783285433461,
            0.2031674267230657,
            0.2334925365383547,
            0.2491470458134026,
            0.2491470458134026,
            0.2334925365383547,
            0.2031674267230657,
            0.1600783285433461,
            0.1069393259953178,
            0.0471753363865132,
        ]
    )
    integral = 0.0
    for i, w in enumerate(weights):
        x = 0.5 * (b - a) * (roots[i] + 1.0) + a
        xpow = 1.0
        for _ in range(p):
            xpow *= x
        integral += xpow / np.expm1(x) * w
    return integral * (b - a) / mu


@vectorize(
    [
        float32(float32, float32, int32),
        float64(float64, float64, int64),
    ],
    target="parallel",
    fastmath=True,
)
def planck_integral(x1, x2, p):
    """Returns the definite integral of the normalized Planck function
    norm * x^p/(exp(x)-1) from x1 to x2, accurate to machine precision
    """
    if x1 > x2:  # assume x2 is the larger in magnitude
        x1, x2 = x2, x1
        fac = -1.0
    else:
        fac = 1.0

    normalized = True
    if normalized:
        fac /= PLANCK_NORM[p - 1]

    cutoff_for_series = 3.0

    if x1 > cutoff_for_series:
        return fac * planck_upper_series(x1, x2, p)
    if x2 < cutoff_for_series:
        return fac * planck_gaussquad(x1, x2, p)
    return fac * (planck_gaussquad(x1, cutoff_for_series, p) + planck_upper_series(cutoff_for_series, x2, p))


def planck_wavelength_integral(min_wavelength_um, max_wavelength_um, temp_K):
    """Returns the fraction of SED energy between two wavelengths specified in microns,
    for a given temperature"""

    # convert to quantity x = h c / (wavelength k_B T)
    const = (h * c / (k_B * u.K * u.um)).cgs.value
    min_wavelength_um = np.array(min_wavelength_um)
    max_wavelength_um = np.array(max_wavelength_um)
    x_max = np.float64(const / (min_wavelength_um * temp_K))
    x_min = np.float64(const / (max_wavelength_um * temp_K))
    return planck_integral(x_min, x_max, 3)
