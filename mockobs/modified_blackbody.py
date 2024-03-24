"""Routines for performing modified blackbody fits to dust emission SEDS and datacubes"""

import numpy as np
from numpy.linalg import pinv
from numba import njit, prange
from scipy.optimize import curve_fit
from astropy import constants, units as u


@njit(fastmath=True, error_model="numpy")
def modified_planck_function(freqs, logtau, beta, T):
    """Modified blackbody function"""
    h_over_kB, h, c, f0 = (
        4.799243073366221e-11,
        6.62607015e-27,
        29979245800.0,
        599584916000.0,
    )
    return (
        10**logtau
        * 2
        * h
        / c**2
        * freqs**3
        * (freqs / f0) ** beta
        / np.expm1(freqs * h_over_kB / T)
    )


@njit(error_model="numpy", parallel=True)
def modified_blackbody_fit_image(image, wavelengths):
    """Fit each pixel in a datacube to a modified blackbody"""
    res = (image.shape[0], image.shape[1])
    params = np.empty((res[0], res[1], 3))
    for i in prange(res[0]):
        for j in range(res[1]):
            # print(i, j)
            params[i, j] = modified_blackbody_fit_gaussnewton(image[i, j], wavelengths)
    return params


@njit(error_model="numpy")
def modified_blackbody_fit_gaussnewton(sed, wavelengths, p0=(0, 1.5, 1.5)):
    """Fits a modified blackbody SED to the provided SED sampled at
    a set of wavelengths using Gauss-Newton iteration
    """
    freqs = 299792458000000.0 / wavelengths

    error = error_best = 1e100
    tol = 0.1
    max_iter = 30

    # preliminary step: assume constant beta and iterate just for logtau and T
    logtau, beta, logT = p0
    params = np.array([logtau, logT])
    i = 0
    while error > tol and i < max_iter:
        logtau, logT = params
        residual = sed - modified_planck_function(freqs, logtau, beta, 10**logT)
        try:
            jac_inv = pinv(
                blackbody_residual_jacobian(freqs, logtau, beta, logT)[:, 0::2]
            )
        except:
            return np.nan * np.ones(3)
        params += jac_inv @ residual
        error = np.abs(params[0] - logtau)
        res_sqr = (residual * residual).sum()
        error_best = min(error_best, res_sqr)
        i += 1
        if i > max_iter:
            return np.nan * np.ones(3)

    params = np.array([logtau, beta, logT])
    tol, i, error, error_best = 1e-3, 0, 1e100, 1e100
    while error > tol and i < max_iter:
        logtau, beta, logT = params
        residual = sed - modified_planck_function(freqs, logtau, beta, 10**logT)
        try:
            jac_inv = pinv(blackbody_residual_jacobian(freqs, logtau, beta, logT))
        except:
            return np.nan * np.ones(3)
        params += jac_inv @ residual
        error = np.abs(params[2] - logT)
        res_sqr = (residual * residual).sum()
        error_best = min(error_best, res_sqr)
        i += 1
        if i > max_iter:
            return np.nan * np.ones(3)

    return params


def modified_blackbody_fit_curvefit(sed, wavelengths, p0=(0, 1.5, 1.5)):
    freqs = (constants.c / (wavelengths * u.si.micron)).cgs.value
    kTconst = constants.h.cgs.value / constants.k_B.cgs.value
    fmin = freqs.min()

    def fitfunc(freq, logtau, beta, T):
        return modified_planck_function(freqs, logtau, beta, T)

    return curve_fit(fitfunc, freqs, sed / sed.min(), p0=p0)[0]


@njit(fastmath=True, error_model="numpy")
def blackbody_residual_jacobian(freqs, logtau, beta, logT):
    """Jacobian for least-squares fit to modified blackbody"""
    h = 6.62607015e-27
    c = 29979245800.0
    f0 = 599584916000.0
    k = 1.380649e-16
    T = 10**logT

    jac = np.empty((freqs.shape[0], 3))
    for i in range(freqs.shape[0]):
        f = freqs[i]
        e_over_kT = f * h / (k * T)
        expfac = np.expm1(e_over_kT)
        fbetafac = f**3 * (f / f0) ** beta
        modbb = 10**logtau * 2 * h / c**2 * fbetafac / expfac
        jac[i, 2] = (
            np.log(10)
            * 10**logtau
            * f
            * fbetafac
            * h**2
            / (c**2 * k * T * (np.cosh(e_over_kT) - 1))
        )
        jac[i, 1] = np.log(f / f0) * modbb
        jac[i, 0] = np.log(10) * modbb
    return jac
